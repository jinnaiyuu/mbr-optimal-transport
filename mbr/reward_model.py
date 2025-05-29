from openai import AzureOpenAI
from huggingface_hub import snapshot_download
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
import llm_blender  # This is not available in the current environment.
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import nn
import torch
import ginza
import spacy
from typing import List

import numpy as np
import math
import os

import nltk

nltk.download("punkt")

HF_READ_TOKEN = os.getenv("HF_READ_TOKEN")
if HF_READ_TOKEN is None:
    print("HF_READ_TOKEN is not set. Please set it in your environment.")



# from tenacity import retry, stop_after_attempt, wait_random_exponential

_CLAIR_PROMPT = """\
You are trying to tell if a candidate caption is describing the same image as a reference set of captions.

Candidate caption:
{answer}

Reference set:
{question}

On a precise scale from 0 to 100, how likely is it that the candidate set is \
describing the same image as the reference set? \
You must by strictly follow this format: \"[[rating]]\", for example: \"Rating: [[50]]\".
"""


class RewardModel:
    def __init__(self, reward_model_id):
        self.get_raw = False
        self.prompt = None

    def get_reward(self, question: str, answer: str, context: str) -> float:
        pass

    def get_rewards(
        self, question: str, answers: List[str], context: str = None
    ) -> List[float]:
        scores = []
        for i in range(len(answers)):
            score = self.get_reward(question, answers[i], context)
            scores.append(score)
        return scores

    def get_pairwise_reward(
        self, question: str, answer: str, compared_answer: str
    ) -> float:
        pass

    def get_pairwise_rewards(self, question: str, answers: List[str]) -> np.ndarray:
        scores_matrix = np.zeros((len(answers), len(answers)))
        for i in range(len(answers)):
            for j in range(len(answers)):
                score = self.get_pairwise_reward(question, answers[i], answers[j])
                scores_matrix[i][j] = score
        return scores_matrix

    def set_get_raw(self, get_raw: bool):
        self.get_raw = get_raw

    def get_prompt(self, question, answer, context=None):
        d = {"question": question, "answer": answer, "context": context}
        return self.prompt.format(**d)

    def set_prompt(self, prompt):
        self.prompt = prompt


class OASST(RewardModel):
    def __init__(self, reward_model_id):
        super().__init__(reward_model_id)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_id, token=HF_READ_TOKEN
        )
        self.reward_model.eval()
        self.reward_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            reward_model_id, token=HF_READ_TOKEN
        )

    def get_reward(self, question, answer, context=None):
        # TODO: Batch operation.
        inputs = self.tokenizer(question, answer, return_tensors="pt").to(self.device)
        outputs = (
            self.reward_model(**inputs).logits[0].cpu().detach().numpy().item()
        )  # TODO: This doesn't work for CPU??
        return outputs


class Pythia(RewardModel):
    def __init__(self, reward_model_id):
        super().__init__(reward_model_id)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_id, token=HF_READ_TOKEN
        )
        self.reward_model.eval()
        self.reward_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            reward_model_id, token=HF_READ_TOKEN
        )
        self.prompt = "<|instruction|> {} <|response|> {}"

    def get_reward(self, question, answer, context=None):
        text = self.prompt.format(question, answer)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.reward_model(**inputs).logits[0].cpu().detach().numpy().item()
        return outputs


class StanfordNLP(RewardModel):
    def __init__(self, reward_model_id):
        super().__init__(reward_model_id)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(reward_model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(reward_model_id).to(
            self.device
        )
        self.model.eval()

    def get_reward(self, question, answer, context=None):
        input_text = "POST: {} \n\n RESPONSE A: {}\n\n RESPONSE B: .\n\n Which response is better? RESPONSE".format(
            question.replace("\n", " "), answer.replace("\n", " ")
        )
        x = self.tokenizer([input_text], return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1
        )
        prob_of_A = torch.exp(outputs.scores[0][:, 71]) / torch.exp(
            outputs.scores[0][:, :]
        ).sum(
            axis=1
        )  # index 71 corresponds to the token for 'A'
        return prob_of_A.cpu().detach().numpy().item()

    def get_pairwise_reward(
        self, question: str, answer: str, compared_answer: str
    ) -> float:
        input_text = "POST: {} \n\n RESPONSE A: {}\n\n RESPONSE B: {}\n\n Which response is better? RESPONSE".format(
            question.replace("\n", " "),
            answer.replace("\n", " "),
            compared_answer.replace("\n", " "),
        )
        x = self.tokenizer([input_text], return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1
        )
        prob_of_A = torch.exp(outputs.scores[0][:, 71]) / torch.exp(
            outputs.scores[0][:, :]
        ).sum(
            axis=1
        )  # index 71 corresponds to the token for 'A'
        return prob_of_A.cpu().detach().numpy().item()


class Eurus(RewardModel):
    def __init__(self, reward_model_id):
        super().__init__(reward_model_id)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
        self.model = AutoModel.from_pretrained(
            reward_model_id, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def get_reward(self, question, answer, context=None):
        input_text = "[INST] {} [\\INST] {}".format(question, answer)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        chosen_reward = self.model(**inputs).item()
        return chosen_reward


class PairLM(RewardModel):
    def __init__(self, reward_model_id):
        super().__init__(reward_model_id)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM")
        self.blender.blender_config.use_tqdm = False

        # This is for pairwise comparison.
        # We don't need self.blender but for backward compatibility, we keep
        # it.
        self.pairrm = DebertaV2PairRM.from_pretrained("llm-blender/PairRM-hf").to(
            self.device
        )
        self.pairrm.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("llm-blender/PairRM-hf")
        self.source_prefix = "<|source|>"
        self.cand1_prefix = "<|candidate1|>"
        self.cand2_prefix = "<|candidate2|>"

    def get_reward(self, question, answer, context=None):
        print("PairLM.get_reward() not implemented.")
        assert False

    def get_rewards(self, question, answers, context=None):
        ranks = self.blender.rank(
            [question], [list(answers)], return_scores=False, batch_size=1
        )
        return (1 - ranks[0]).tolist()

    def get_winratio(self, question, answer, compared_answers):
        assert isinstance(question, str)
        assert isinstance(answer, str)
        assert isinstance(compared_answers, list)
        assert isinstance(compared_answers[0], str)

        wins = 0
        cs = list(compared_answers)
        ncs = len(cs)
        pairs = [[answer, cs[i]] for i in range(ncs)]
        ranks = self.blender.rank(
            [question] * ncs, pairs, return_scores=False, batch_size=16
        )

        wins = (ranks[:, 0] < ranks[:, 1]).sum() / ncs

        return wins

    def tokenize_pair(
        self,
        sources: List[str],
        candidate1s: List[str],
        candidate2s: List[str],
        source_max_length=1224,
        candidate_max_length=412,
    ):
        ids = []
        assert len(sources) == len(candidate1s) == len(candidate2s)
        max_length = source_max_length + 2 * candidate_max_length
        for i in range(len(sources)):
            source_ids = self.tokenizer.encode(
                self.source_prefix + sources[i],
                max_length=source_max_length,
                truncation=True,
            )
            candidate_max_length = (max_length - len(source_ids)) // 2
            candidate1_ids = self.tokenizer.encode(
                self.cand1_prefix + candidate1s[i],
                max_length=candidate_max_length,
                truncation=True,
            )
            candidate2_ids = self.tokenizer.encode(
                self.cand2_prefix + candidate2s[i],
                max_length=candidate_max_length,
                truncation=True,
            )
            ids.append(source_ids + candidate1_ids + candidate2_ids)
        encodings = self.tokenizer.pad(
            {"input_ids": ids},
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
        )
        return encodings

    def get_pairwise_reward(
        self, question: str, answer: str, compared_answer: str
    ) -> float:
        encodings = self.tokenize_pair([question], [answer], [compared_answer])
        encodings = {k: v.to(self.pairrm.device) for k, v in encodings.items()}
        outputs = self.pairrm(**encodings)
        logits = outputs.logits.tolist()
        return logits[0]

    # def get_pairwise_rewards(self, question: str, answers: List[str]) -> np.ndarray:
    #     logits_matrix = []
    #     for answer in answers:
    #         encodings = self.tokenize_pair([question] * len(answers), [answer] * len(answers), answers)
    #         encodings = {k:v.to(self.pairrm.device) for k,v in encodings.items()}
    #         outputs = self.pairrm(**encodings)
    #         logits = outputs.logits.tolist()
    #         logits_matrix.append(logits)
    #     return np.array(logits_matrix)


class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        self.config.n_embd = (
            self.config.hidden_size
            if hasattr(self.config, "hidden_size")
            else self.config.n_embd
        )
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        return scores


class Starling(RewardModel):
    def __init__(self, reward_model_id):
        super().__init__(reward_model_id)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("initializing Starling...")
        self.reward_model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf").to(
            self.device
        )
        self.reward_tokenizer = self.reward_model.tokenizer
        self.reward_tokenizer.truncation_side = "left"

        directory = snapshot_download("berkeley-nest/Starling-RM-7B-alpha")
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith("model.bin"):
                checkpoint = os.path.join(directory, fpath)
                break

        self.reward_model.load_state_dict(torch.load(checkpoint), strict=False)
        self.reward_model.eval().requires_grad_(False)

    def get_reward_(self, samples):
        """samples: List[str]"""
        input_ids = []
        attention_masks = []
        encodings_dict = self.reward_tokenizer(
            samples,
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        input_ids = encodings_dict["input_ids"]
        attention_masks = encodings_dict["attention_mask"]
        mbs = 4
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            rewards = self.reward_model(
                input_ids=input_ids[i * mbs : (i + 1) * mbs],
                attention_mask=attention_masks[i * mbs : (i + 1) * mbs],
            )
            out.extend(rewards)
        return torch.hstack(out)

    def get_reward(self, question, answer, context=None):
        sequences = ["<s>[INST] {} </s> [/INST] {}</s>".format(question, answer)]
        return self.get_reward_(sequences).cpu().detach().numpy().item()

    def get_rewards(self, question, answers, context=None):
        rewards = []
        for answer in answers:
            rewards.append(self.get_reward(question, answer))
        return rewards


class GemmaJudge(RewardModel):
    def __init__(self, reward_model_id, load_in_4bit=False, load_in_8bit=False):
        super().__init__(reward_model_id)
        self.reward_model_id = reward_model_id

        if load_in_4bit or load_in_8bit:
            self.reward_model = AutoModelForCausalLM.from_pretrained(
                reward_model_id,
                trust_remote_code=True,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                device_map="auto",
            )
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.reward_model = AutoModelForCausalLM.from_pretrained(
                reward_model_id, torch_dtype=torch.bfloat16
            )
            self.reward_model.to(device)

        self.reward_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(reward_model_id)

        self.prompt = """**Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".**

[Question]
{question}
[The Start of Assistant’s Answer]
{answer}
[The End of Assistant’s Answer]
"""

    def get_reward(self, question, answer, context=None):
        response = str(self.get_response(question, answer))
        assert isinstance(response, str)

        try:
            score = response.split("[[")[1].split("]]")[0]
            score_ = int(score)
        except BaseException:
            score_ = -1

        if self.get_raw:
            return score_, response
        else:
            return score_

    def get_response(self, question, answer, context=None) -> str:
        if context is None:
            prompt = self.get_prompt(question, answer)
        else:
            prompt = self.get_prompt(question, answer, context)

        # if not ('gemma' in self.reward_model_id):
        msg = [{"role": "user", "content": prompt}]
        in_prompt = self.tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )

        input_ids = self.tokenizer(in_prompt, return_tensors="pt").to(
            self.reward_model.device
        )

        outputs_id = self.reward_model.generate(
            **input_ids, do_sample=True, max_new_tokens=1024, temperature=0.1
        )
        # outputs_id = self.reward_model.generate(**input_ids, do_sample=True, max_new_tokens=64, temperature=0.1)

        # if 'gemma' in self.reward_model_id:
        #     input_length = input_ids['input_ids'].shape[1]
        #     output_text = self.tokenizer.decode(outputs_id[0][input_length:], skip_special_tokens=True)
        # else:
        input_length = input_ids["input_ids"].shape[1]
        output_text = self.tokenizer.decode(
            outputs_id[0][input_length:], skip_special_tokens=True
        )
        print("GemmaJudge: in_prompt=", in_prompt)
        print("GemmaJudge: output_text=", output_text)
        return output_text


class GPT4Eval(RewardModel):
    def __init__(self, reward_model_id):
        endpoint = "https://ailab-rl-openai-east-us.openai.azure.com/"
        api_key = "3b9c5738937b42a7a35354f65dd57b02"
        self.model_name = "gpt-4o"

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2023-12-01-preview",
        )

        super().__init__(reward_model_id)

        self.prompt = """**Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".**

[Question]
{question}
[The Start of Assistant’s Answer]
{answer}
[The End of Assistant’s Answer]
"""

        self.pairwise_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.

[User Question]
{question}

[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
"""

    # def completion_with_backoff(**kwargs):
    #     return openai.ChatCompletion.create(**kwargs)

    def get_reward(self, question, answer, context=None):
        try:
            response = str(self.get_response(question, answer))
            assert isinstance(response, str)

            score = response.split("[[")[1].split("]]")[0]
            score_ = int(score)
        except BaseException:
            score_ = -1

        if self.get_raw:
            return score_, response
        else:
            return score_

    def get_response(self, question, answer, context=None) -> str:
        output_text = self.gpt4eval(question, answer, context)
        # output_text.replace(prompt, "")
        return output_text

    def gpt4eval(self, question, answer, context=None):
        if context is None:
            prompt = self.get_prompt(question, answer)
        else:
            prompt = self.get_prompt(question, answer, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                # model='gpt-35-turbo',
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                frequency_penalty=0,
                presence_penalty=0,
            )
        except Exception as e:
            print(e)
            return "[[-1]]"
            # assert False

        return response.choices[0].message.content

    def set_model(self, model_name):
        self.model_name = model_name

    def get_pairwise_reward(
        self, question: str, answer: str, compared_answer: str
    ) -> float:
        d = {"question": question, "answer_a": answer, "answer_b": compared_answer}
        prompt = self.pairwise_prompt.format(**d)

        response = self.client.chat.completions.create(
            model=self.model_name,
            # model='gpt-35-turbo',
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            frequency_penalty=0,
            presence_penalty=0,
        )

        response_text = response.choices[0].message.content

        if "[[A]]" in response_text:
            return 1
        elif "[[B]]" in response_text:
            return 0
        elif "[[C]]" in response_text:
            return 0.5
        else:
            return -1


class NumChars(RewardModel):
    def __init__(self, reward_model_id):
        super().__init__(reward_model_id)

    def get_reward(self, question, answer, context=None):
        return len(answer)

    def get_pairwise_reward(
        self, question: str, answer: str, compared_answer: str
    ) -> float:
        return len(answer) - len(compared_answer)

    def get_rewards(self, question, answers, context=None):
        return [len(answer) for answer in answers]

    def get_pairwise_rewards(self, question: str, answers: List[str]) -> np.ndarray:
        return np.array(
            [
                [len(answer) - len(compared_answer) for compared_answer in answers]
                for answer in answers
            ]
        )


class NumSentences(RewardModel):
    def __init__(self, reward_model_id, lang="en"):
        super().__init__(reward_model_id)
        self.lang = lang

        if self.lang == "ja":
            self.nlp = spacy.load("ja_ginza")
            self.nlp.disable_pipes(
                [
                    "tok2vec",
                    "parser",
                    "ner",
                    "morphologizer",
                    "compound_splitter",
                    "bunsetu_recognizer",
                ]
            )
            self.nlp.add_pipe("sentencizer")

    def get_reward(self, question, answer, context=None):
        return len(self.text2sentences(answer))

    def get_rewards(self, question, answers, context=None):
        return [len(self.text2sentences(answer)) for answer in answers]

    def text2sentences(self, text):
        if self.lang == "en":
            return nltk.sent_tokenize(text)
        elif self.lang == "de":
            return nltk.sent_tokenize(text, language="german")
        elif self.lang == "ja":
            doc = self.nlp(text)
            return [sent.text for sent in doc.sents]
        else:
            raise ValueError(f"Invalid language: {self.lang}")


def load_reward_model(reward_model_id):
    if reward_model_id == "OpenAssistant/reward-model-deberta-v3-large-v2":
        return OASST(reward_model_id)
    elif "ddyuudd/alpaca" in reward_model_id:
        return OASST(reward_model_id)
    elif "ddyuudd/pythia" in reward_model_id:
        return Pythia(reward_model_id)
    elif "ddyuudd/hh-rlhf-pythia" in reward_model_id:
        rm = Pythia(reward_model_id)
        rm.set_prompt = "Human: {} Assistant: {}"
        return rm
    elif "stanfordnlp/SteamSHP" in reward_model_id:
        return StanfordNLP(reward_model_id)
    elif reward_model_id == "llm-blender/PairRM":
        return PairLM(reward_model_id)
    elif "berkeley-nest" in reward_model_id:
        return Starling(reward_model_id)
    elif "openbmb/Eurus-RM-7b" in reward_model_id:
        return Eurus(reward_model_id)
    elif "google/gemma" in reward_model_id:
        return GemmaJudge(reward_model_id)
    elif "jcm-gemma27" in reward_model_id:
        rm = GemmaJudge("google/gemma-2-27b-it")
        with open(os.path.join("./prompts", "jcm_eval.txt"), "r") as f:
            prompt = f.read()
        rm.set_prompt(prompt)
        return rm
    elif "jcm-gemma" in reward_model_id:
        rm = GemmaJudge("google/gemma-2-9b-it")
        with open(os.path.join("./prompts", "jcm_eval.txt"), "r") as f:
            prompt = f.read()
        rm.set_prompt(prompt)
        return rm
    elif "gpt4" == reward_model_id:
        return GPT4Eval(reward_model_id)
    elif "gpt35" == reward_model_id:
        rm = GPT4Eval(reward_model_id)
        rm.set_model("gpt-35-turbo")
        return rm
    elif "gpt4mini" == reward_model_id:
        rm = GPT4Eval(reward_model_id)
        rm.set_model("gpt-4o-mini")
        return rm
    elif "length" == reward_model_id:
        return NumChars(reward_model_id)
    elif "nsentences" in reward_model_id:
        return NumSentences("nsentences", lang=reward_model_id.split("-")[1])
    else:
        print("ERROR: Invalid reward_model_id", reward_model_id)
        raise ValueError("Invalid reward_model_id")


# def load_reward_model(reward_model_id):

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     if reward_model_id == 'OpenAssistant/reward-model-deberta-v3-large-v2':
#         reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id)
#         reward_model.eval()
#         reward_model.to(device)
#         tokenizer = AutoTokenizer.from_pretrained(reward_model_id)

#         def get_reward(question, answer):
#             # TODO: Batch operation.
#             inputs = tokenizer(question, answer, return_tensors='pt').to(device)
#             outputs = reward_model(**inputs).logits[0].cpu().detach().numpy().item()
#             return outputs
#     elif reward_model_id == 'stanfordnlp/SteamSHP-flan-t5-xl':

#         tokenizer = T5Tokenizer.from_pretrained(reward_model_id)
#         model = T5ForConditionalGeneration.from_pretrained(reward_model_id).to(device)
#         model.eval()
#         def get_reward(question, answer):
#             input_text = "POST: {} \n\n RESPONSE A: {}\n\n RESPONSE B: .\n\n Which response is better? RESPONSE".format(question.replace('\n', ' '), answer.replace('\n', ' '))
#             x = tokenizer([input_text], return_tensors='pt').input_ids.to(device)
#             outputs = model.generate(x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
#             prob_of_A = torch.exp(outputs.scores[0][:, 71]) / torch.exp(outputs.scores[0][:,:]).sum(axis=1).item() # index 71 corresponds to the token for 'A'
#             return prob_of_A

#     return get_reward
