from typing import List, Optional, Union, Tuple, Dict, Any

import torch
from torch import nn

from transformers import (
    BartForConditionalGeneration, 
    T5ForConditionalGeneration,
    LogitsProcessorList, 
    LogitsProcessor, 
    PreTrainedTokenizer, 
    TemperatureLogitsWarper, 
    StoppingCriteriaList
)
from transformers.generation_utils import GenerationMixin, BeamSampleOutput, BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput
from transformers.file_utils import ModelOutput
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer


class SelfDebiasingLogitsProcessor(LogitsProcessor):
    """This class represents a logits processor that applies self-debiasing."""

    def __init__(self, num_debiasing_prefixes: int, decay_constant: float = 50, epsilon: float = 0.01, debug: bool = False,
                 tokenizer: Optional[PreTrainedTokenizer] = None, 
                 reg_temperature: float = 1.0, bias_temperature: float = 1.0):
        """
        :param num_debiasing_prefixes: the number of debiasing prefixes used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :param debug: whether to print additional debugging output
        :param tokenizer: a tokenizer used to print debugging output
        """
        assert not debug or tokenizer, "If debug=True, a tokenizer must be passed to SelfDebiasingLogitsProcessor()"
        self.num_debiasing_prefixes = num_debiasing_prefixes
        self.decay_constant = decay_constant
        self.epsilon = epsilon
        self.debug = debug
        self.tokenizer = tokenizer
        self.reg_temperature = reg_temperature
        self.bias_temperature = bias_temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = scores.shape[0] // (1 + self.num_debiasing_prefixes)
        regular_sentence_indices = range(batch_size)
        for regular_sentence_idx in regular_sentence_indices:
            if self.debug:
                print(f'== Generated Text ==\n', 
                      self.tokenizer.decode(input_ids[regular_sentence_idx]))
            bias_indices = self._get_bias_indices(regular_sentence_idx, batch_size)
            if bias_indices:
                self._debias_scores(scores, regular_sentence_idx, bias_indices)
        return scores

    def _get_bias_indices(self, regular_sentence_idx: int, batch_size: int) -> List[int]:
        """Returns the indices of all self-debiasing inputs for a regular input"""
        return [regular_sentence_idx + (prefix_idx + 1) * batch_size for prefix_idx in range(self.num_debiasing_prefixes)]

    def _debias_scores(self, scores: torch.FloatTensor, regular_sent_idx: int, bias_indices: List[int]) -> None:
        """Partially debiases the given scores considering a single sentence and the corresponding self-debiasing inputs"""
        scores[regular_sent_idx] = scores[regular_sent_idx] / self.reg_temperature
        
        logits_biased = [scores[bias_idx] / self.bias_temperature for bias_idx in bias_indices]

        mask = self._generate_decay_mask(scores[regular_sent_idx], logits_biased)
        logits_regular_debiased = self._apply_decay_mask(scores[regular_sent_idx], mask)
        if self.debug:
            print(f'Top 5 predictions (regular): {self._get_most_likely_tokens(logits_regular_debiased, k=5)}\n')
        scores[regular_sent_idx] = torch.log(logits_regular_debiased)

        for debiasing_sent_idx in bias_indices:
            scores[debiasing_sent_idx] = scores[regular_sent_idx]

    def _apply_decay_mask(self, logits: torch.Tensor, decay_mask: torch.Tensor) -> torch.Tensor:
        """Applies exponential decay to a tensor of logits"""
        probabilities = logits.softmax(dim=-1)
        decay_mask = torch.exp(- decay_mask * self.decay_constant)
        decay_mask = torch.max(decay_mask, torch.tensor([self.epsilon], device=decay_mask.device))
        if self.debug:
            print(f'== After Debiasing ==\n'
                  f'Top 5 masked tokens: {self._get_most_likely_tokens(decay_mask, k=5, largest=False)}')  
        probabilities = probabilities * decay_mask
        probabilities = probabilities / probabilities.sum(dim=-1)
        return probabilities

    def _generate_decay_mask(self, logits_regular: torch.FloatTensor, logits_biased_list: List[torch.FloatTensor]) -> torch.Tensor:
        """Computes the alpha values (see paper) for each token and stores them in a mask tensor"""
        p_regular = logits_regular.softmax(dim=-1)
        p_biased = None

        for logits_biased in logits_biased_list:
            if p_biased is None:
                p_biased = logits_biased.softmax(dim=-1)
            else:
                p_biased = torch.max(p_biased, logits_biased.softmax(dim=-1))

        if self.debug:
            print(f'== Before Debiasing ==\n'
                  f'Top 5 predictions (regular): {self._get_most_likely_tokens(p_regular, k=5)}\n'
                  f'Top 5 predictions (biased): {self._get_most_likely_tokens(p_biased, k=5)}')

        mask = torch.max(p_biased - p_regular, torch.tensor([0.], device=p_regular.device))
        return mask

    def _get_most_likely_tokens(self, probabilities_tensor: torch.Tensor, k: int, largest: bool = True) -> List[Tuple[str, float]]:
        """Returns the most likely tokens according to a tensor of probabilities"""
        assert len(probabilities_tensor.shape) == 1
        values, indices = torch.topk(probabilities_tensor, k=k, dim=-1, largest=largest)
        tokens = self.tokenizer.convert_ids_to_tokens(indices)
        return list(zip(tokens, [pv.item() for pv in values]))


class SelfDebiasingGenerationMixin(GenerationMixin):
    
    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
            # =========================
            # BEGIN MODIFICATIONS
            if model_kwargs.get("decoder_attention_mask") is not None:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask.index_select(0, expanded_return_idx)
            # =========================
            # END MODIFICATIONS
        return input_ids, model_kwargs

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        # =========================
        # BEGIN MODIFICATIONS
        else:
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))], dim=-1
                )
        # END MODIFICATIONS
        # =========================

        return model_kwargs
    
    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[BeamSampleOutput, torch.LongTensor]:
        """
        This is a verbatim copy of the original implementation by Hugging Face, with a single modification to ensure that a text and all
        corresponding self-debiasing inputs always chose the same token to generate next. This modification is enclosed by the texts
        "BEGIN MODIFICATIONS" and "END MODIFICATIONS", respectively.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (logits_warper(input_ids, next_token_scores_processed),)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = nn.functional.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # =========================
            # BEGIN MODIFICATIONS
            # the following modification to the sample method is necessary to ensure that each debiasing sentence is continued in the same
            # way as the original sentence
            if self.logits_processor is not None:
                n_regular_sentences = next_tokens.shape[0] // (1 + self.logits_processor.num_debiasing_prefixes)
                regular_sentence_indices = range(n_regular_sentences)
                for regular_sentence_idx in regular_sentence_indices:
                    debiasing_sentence_indices = self.logits_processor._get_bias_indices(regular_sentence_idx, n_regular_sentences)
                    for debiasing_sentence_idx in debiasing_sentence_indices:
                        next_tokens[debiasing_sentence_idx] = next_tokens[regular_sentence_idx]
                        next_indices[debiasing_sentence_idx] = next_indices[regular_sentence_idx]
                        next_token_scores[debiasing_sentence_idx] = next_token_scores[regular_sentence_idx]
            # END MODIFICATIONS
            # =========================

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            else:
                num_return_sequences = beam_scorer.num_beam_hyps_to_keep
                # return only as many indices as sequences
                beam_indices = tuple(
                    (beam_indices[i * num_beams : i * num_beams + num_return_sequences] for i in range(batch_size))
                )
                beam_indices = sum(beam_indices, ())

            if self.config.is_encoder_decoder:
                return BeamSampleEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSampleDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]  


class SelfDebiasingBartForConditionalGeneration(BartForConditionalGeneration, SelfDebiasingGenerationMixin):
    """
    This class represents a regular BartForConditionalGeneration that additionally has the capacity to perform self-debiasing. For self-debiasing, the
    init_logits_processor function must be called. Otherwise, this model just performs regular conditional generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits_processor = None  # type: Optional[SelfDebiasingLogitsProcessor]

    def init_logits_processor(self, *args, **kwargs):
        """Initialize the logits processor. For a list of arguments, see the self-debiasing logit processor's init function."""
        self.logits_processor = SelfDebiasingLogitsProcessor(*args, **kwargs)

    def _get_logits_processor(self, *args, **kwargs) -> LogitsProcessorList:
        logits_processor = super()._get_logits_processor(*args, **kwargs)
        if self.logits_processor is not None:
            logits_processor.append(self.logits_processor)
        return logits_processor

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        # =========================
        # BEGIN MODIFICATIONS
        decoder_attention_mask=None,
        # =========================
        # END MODIFICATIONS
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            # =========================
            # BEGIN MODIFICATIONS
            "decoder_attention_mask": decoder_attention_mask,
            # END MODIFICATIONS
            # =========================
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }


class SelfDebiasingT5ForConditionalGeneration(T5ForConditionalGeneration, SelfDebiasingGenerationMixin):
    """
    This class represents a regular T5ForConditionalGeneration that additionally has the capacity to perform self-debiasing. For self-debiasing, the
    init_logits_processor function must be called. Otherwise, this model just performs regular conditional generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits_processor = None  # type: Optional[SelfDebiasingLogitsProcessor]

    def init_logits_processor(self, *args, **kwargs):
        """Initialize the logits processor. For a list of arguments, see the self-debiasing logit processor's init function."""
        self.logits_processor = SelfDebiasingLogitsProcessor(*args, **kwargs)

    def _get_logits_processor(self, *args, **kwargs) -> LogitsProcessorList:
        logits_processor = super()._get_logits_processor(*args, **kwargs)
        if self.logits_processor is not None:
            logits_processor.append(self.logits_processor)
        return logits_processor

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        # =========================
        # BEGIN MODIFICATIONS
        decoder_attention_mask=None,
        # =========================
        # END MODIFICATIONS
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            # =========================
            # BEGIN MODIFICATIONS
            "decoder_attention_mask": decoder_attention_mask,
            # END MODIFICATIONS
            # =========================
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
