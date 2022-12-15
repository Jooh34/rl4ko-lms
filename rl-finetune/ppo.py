import torch
import torch.nn as nn
from transformers import LogitsProcessorList, StoppingCriteriaList
from transformers.generation_beam_search import BeamSearchScorer
from torch.distributions import Categorical
import os, sys
from typing import List
import numpy as np
import wandb

cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_dir, '../'))
from common.rouge import RougeScorer, FakeTokenizer


device = 'cuda'
# class BartState:
#     def __init__(self, input_ids, attention_mask, decoder_input_ids):
#         self.input_ids=input_ids,
#         self.attention_mask=attention_mask,
#         self.decoder_input_ids=decoder_input_ids

class RolloutBuffer:
    def __init__(self):
        # below 3 is state.
        self.input_ids = []
        self.attention_mask = []
        self.decoder_input_ids = []

        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
    
    def clear(self):
        del self.input_ids[:]
        del self.attention_mask[:]
        del self.decoder_input_ids[:]

        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]

class PPO(nn.Module):
    def __init__(self, cfg, actor_model, old_model, tokenizer, optimizer, scheduler=None, K_epochs=80, eps_clip=0.2):
        super(PPO, self).__init__()
        self.cfg = cfg
        self.actor_model = actor_model
        self.old_model = old_model

        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.buffer = RolloutBuffer()
        self.K_epochs = K_epochs

        self.token_rouge_scorer = RougeScorer(FakeTokenizer())
        self.old_model.load_state_dict(self.actor_model.state_dict())
        self.old_model.eval()

        self.eps_clip = eps_clip
        
    def forward(self):
        raise NotImplementedError

    def save_one_episode(self, input_ids, attention_mask, label_ids, calc_values=False):
        batch_size = len(input_ids)

        encoder = self.old_model.get_encoder()
        output_hidden_states = True
        output_attentions = self.old_model.config.output_attentions

        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        model_kwargs = {"encoder_outputs": encoder_outputs}
        # print('input_ids : ', input_ids)
        # print('attention_mask : ', attention_mask)
        # print('encoder : ', model_kwargs["encoder_outputs"].get("hidden_states") )

        # decoder_input_ids : [B, 1]
        decoder_input_ids = torch.ones_like(input_ids)[:, :1] * self.actor_model.config.decoder_start_token_id

        # sample for reinforcement learning
        if self.cfg['use_beamsearch']:
            sample_output = self.actor_model.generate(
                input_ids,
                max_length=512,
                num_beams=3,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            # beam_scorer = BeamSearchScorer(
            #     batch_size=1,
            #     num_beams=5,
            #     device='cuda',
            #     max_length=512
            # )

            # sample_output = self.actor_model.beam_search(
            #     input_ids,
            #     beam_scorer,
            #     max_length=512,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     **model_kwargs
            # )
        else:
            sample_output = self.actor_model.sample(
                decoder_input_ids,
                max_length=512,
                return_dict_in_generate=True,
                output_scores=True,
                **model_kwargs
            )
        '''
        SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
        '''
        # scores : [B, S, V]
        scores = sample_output.scores
        sequences = sample_output.sequences
        generated = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
        print('generated : ', generated)
        len_token = len(sample_output.scores)

       
        for B in range(batch_size):
            # calculate reward for batch B
            seq = sequences[B]
            lab = label_ids[B]
            lab_cpu = lab.clone().detach().cpu().tolist()
            rouge_score = self.token_rouge_scorer.score(seq.clone().detach().cpu().tolist(), lab_cpu)
            reward = rouge_score['rouge1'].fmeasure + rouge_score['rouge2'].fmeasure + rouge_score['rougeL'].fmeasure
            
            decoder_input_ids_state = [self.actor_model.config.decoder_start_token_id] # start of sentence
            
            for ix in range(len_token):
                probs = nn.functional.softmax(scores[ix][B], dim=-1)
                dist = Categorical(probs)
                action = sequences[B][ix+1]
                action_logprob = dist.log_prob(action)

                if self.cfg['calc_values']:
                    rouge_score = self.token_rouge_scorer.score(decoder_input_ids_state, lab_cpu)
                    value = rouge_score['rouge1'].fmeasure + rouge_score['rouge2'].fmeasure + rouge_score['rougeL'].fmeasure
                    self.buffer.values.append(torch.tensor(value).cpu())

                self.buffer.decoder_input_ids.append(decoder_input_ids_state.copy())
                
                self.buffer.input_ids.append(input_ids[B].detach().cpu())
                self.buffer.attention_mask.append(attention_mask[B].detach().cpu())

                self.buffer.actions.append(action.detach().cpu())
                self.buffer.logprobs.append(action_logprob.detach().cpu())
                self.buffer.rewards.append(torch.tensor(reward).cpu())

                decoder_input_ids_state.append(action.detach().cpu().tolist())

                if action == self.tokenizer.eos_token_id: # ignore padding after eos
                    break
            
            self.buffer.values.append(torch.tensor(reward).cpu())
                
            
    def evaluate(self, old_input_ids, old_attention_mask, old_decoder_input_ids, old_action):
        encoder = self.actor_model.get_encoder()
        output_hidden_states = True
        output_attentions = self.actor_model.config.output_attentions

        encoder_outputs = encoder(
            input_ids=old_input_ids,
            attention_mask=old_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        model_kwargs = {"encoder_outputs": encoder_outputs}
        # encoder_outputs[0].register_hook(lambda x: print('loss-encoder_outputs : ',x))  

        action_logprobs = torch.zeros(1, 0).to(device)
        dist_entropies = torch.zeros(1, 0).to(device)

        for i,decoder_input_ids in enumerate(old_decoder_input_ids):
            input_ids = torch.tensor(decoder_input_ids).unsqueeze(dim=0).to(device)
            model_inputs = self.actor_model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self.actor_model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            next_token_logits = outputs.logits[:, -1, :]
            
            # pre-process distribution
            logits_processor = LogitsProcessorList()
            logits_warper = LogitsProcessorList()
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            if len(old_decoder_input_ids) == 1:
                print(next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            dist = Categorical(probs)
            action = old_action[i]
            
            action_logprob = dist.log_prob(action)
            dist_entropy = dist.entropy()

            action_logprobs = torch.cat([action_logprobs, action_logprob.unsqueeze(dim=0)], dim=1)
            dist_entropies = torch.cat([dist_entropies, dist_entropy.unsqueeze(dim=0)], dim=1)

            model_kwargs = self.actor_model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.actor_model.config.is_encoder_decoder
            )

        # action_logprobs.register_hook(lambda x: print('action_logprobs hook : ',x))  
        # dist_entropies.register_hook(lambda x: print('dist_entropies hook : ',x))  
        return action_logprobs, dist_entropies

    @staticmethod
    def _normalize(adv: torch.Tensor):
        if len(adv) == 1:
            return adv
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def update(self, ):
        # go train mode
        self.actor_model.train()

        old_input_ids = torch.stack(self.buffer.input_ids, dim=0).detach().to(device)
        old_attention_mask = torch.stack(self.buffer.attention_mask, dim=0).detach().to(device)
        old_decoder_input_ids = np.array(self.buffer.decoder_input_ids)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)
        rewards = torch.stack(self.buffer.rewards, dim=0).detach().to(device)
        if self.cfg['calc_values']:
            values = torch.stack(self.buffer.values, dim=0).detach().to(device)

        for k_epoch in range(self.K_epochs):
            #TODO: batch implemetation 
            new_logprobs, dist_entropy = self.evaluate(old_input_ids[0:1], old_attention_mask[0:1], old_decoder_input_ids, old_actions) 

            ratios = torch.exp(new_logprobs - old_logprobs)

            # calculate advantage function
            # if len(values) == 1:
            #     advantages = values
            # else:
            #     advantages = values[1:]-values[:(len(values)-1)]

            # advantages = rewards - values[:(len(values)-1)]
            advantages = values[1:]
            
            sampled_normalized_advantage = self._normalize(advantages)

            # Finding Surrogate Loss
            surr1 = ratios * sampled_normalized_advantage
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * sampled_normalized_advantage

            policy_reward = torch.min(surr1, surr2).mean()
            entropy_bonus = dist_entropy.mean()
            loss = -(policy_reward + 0.01*entropy_bonus)
            # loss.register_hook(lambda x: print('loss-hook : ',x))  
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            wandb.log({"lr": self.optimizer.param_groups[0]["lr"]})

        # Copy new weights into old policy
        self.old_model.load_state_dict(self.actor_model.state_dict())

        # clear buffer
        self.buffer.clear()
        