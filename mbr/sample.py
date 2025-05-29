import os

import numpy as np
from tqdm import tqdm
import pandas as pd
import csv

from transformers import set_seed
import torch

from parser import get_mbr_parser
from utils import load_model, load_dataset, load_kwargs, StoppingCriteriaSub
from utils import sample_dir, prompt_dir

import boto3

# Length penalty has no effect for sampling in this codebase.


def compute_probability_s2s(sample_output):
    """
    This compute_prob function is compatible with seq2seq models.
    Doesn't work on language models.
    """
    bsz = sample_output.sequences.shape[0]
    probs = np.array([1.0] * bsz)
    # terms = [False] * bsz
    for i in range(len(sample_output.scores)):
        p = np.array([1.0] * bsz)
        for b in range(bsz):
            if hasattr(tokenizer, "pad_token_id"):
                if sample_output.sequences[b][i + 1] == tokenizer.pad_token_id:
                    continue
            log_probs = torch.nn.functional.log_softmax(
                sample_output.scores[i][b], dim=-1
            )
            p[b] = torch.exp(log_probs[sample_output.sequences[b][i + 1]])
        probs *= p
        # print('p=', p)
    return probs


def compute_probability_lm(model, outputs):
    """
    This compute_prob function is compatible with langauge models.
    Doesn't work on seq2seq models.
    """

    # transition_scores = model.compute_transition_scores(
    #     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
    # )
    transition_scores = (
        model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        .cpu()
        .to(torch.float32)
    )

    seq_prob = torch.ones(transition_scores.shape[0]).to(torch.float32)
    for i in range(transition_scores.shape[1]):
        seq_prob *= np.exp(transition_scores[:, i])

    return seq_prob.numpy()


def get_texts(tokenizer, outputs, input_length):
    """
    This function is only compatible with langauge models. not for seq2seq
    """
    bsz = outputs.sequences.shape[0]
    output_texts = []
    for b in range(bsz):
        output_text = tokenizer.decode(
            outputs.sequences[b][input_length:], skip_special_tokens=True
        )
        output_texts.append(output_text)
    return output_texts


def sample(
    dataset,
    tokenizer,
    model,
    src_lines,
    torch_device,
    n_lines,
    start_iter,
    n_samples,
    bsz,
    temperature,
    eps,
    topk,
    topp,
    do_sample,
    diversity_penalty,
    prompt,
    stop_tokens,
    model_n,
    discard_prob=False,
    max_new_tokens=None,
):
    n_batches = n_samples // bsz

    if do_sample == 0:
        if n_batches > 1:
            print("n_batches must be 1 for beam search. Setting n_batches to 1.")
        n_batches = 1
        if diversity_penalty < 0.000001:
            print("Running beam search as diversity penalty is zero.")
    elif do_sample < 0:
        if n_batches > 1:
            print("n_batches must be 1 for beam search. Setting n_batches to 1.")
        n_batches = 1
        print("Running beam search as do_sample is negative.")
    else:
        # Running sampling algorithm.
        pass

    os.makedirs(os.path.join(sample_dir, dataset, model_n), exist_ok=True)
    client = boto3.client("s3")

    # TODO: We should be able to override this with CLI.
    model_kwargs = load_kwargs(dataset)
    if max_new_tokens > 0:
        model_kwargs["max_new_tokens"] = max_new_tokens

    # if 'wmt19' in dataset:
    #     model_kwargs = WMT19_KWARGS
    # elif (dataset == 'xsum'):
    #     model_kwargs = XSUM_KWARGS
    # elif dataset == "samsum":
    #     model_kwargs = SAMSUM_KWARGS
    # elif dataset == 'cnndm':
    #     model_kwargs = CNN_KWARGS
    # elif (dataset == 'nocaps') or ('mscoco' in dataset):
    #     model_kwargs = CAPTION_KWARGS
    # else:
    #     assert False

    # if 'mscoco' in dataset:
    #     mscoco_df = pd.read_csv('experiments/mscoco.csv')

    for sample_id in tqdm(range(start_iter, n_lines)):
        if sample_id > len(src_lines):
            break
        # if (dataset == 'nocaps') or (dataset == 'mscoco-ft'):
        #     model_inputs = tokenizer(src_lines[sample_id], return_tensors="pt").to(model.device)
        #     # model_inputs = tokenizer(images=src_lines[sample_id], return_tensors="pt").to(model.device, torch.float16)
        #     # image = src_lines[sample_id]['image']
        #     # model_inputs = tokenizer(images=image, return_tensors="pt").to(model.device, torch.float16)
        # else:
        #     model_inputs = tokenizer(src_lines[sample_id], return_tensors='pt', truncation=True).to(torch_device)

        # TODO: These prompting desing should be put in a separate function.
        # TODO: Prompts are depedent to both the model and the dataset.  This
        # makes refactoring tricky...
        if model_n in ["gpt-4o", "gpt-4o-mini"]:
            if "[[QUESTION]]" in prompt:
                instruction = prompt.replace("[[QUESTION]]", src_lines[sample_id])
            else:
                instruction = src_lines[sample_id]
            model.set_temperature(temperature)

            rows = []
            for i in range(n_samples):
                response = model.get_response(instruction)
                rows.append((response, 0.0))

            df = pd.DataFrame(rows, columns=["text", "probability"])

            filename = "{:04}_eps-{:.2f}_topk-{:02d}_topp-{:.2f}_tmp-{:.2f}".format(
                sample_id, eps, topk, topp, temperature
            )
            outfilepath = os.path.join(sample_dir, dataset, model_n, filename)
            df.to_csv(
                outfilepath,
                sep=",",
                escapechar="\\",
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                float_format="%.32f",
            )
            s3_path = os.path.join(
                "fairseq", "mbr", "samples", dataset, model_n, filename
            )
            client.upload_file(outfilepath, "ailab-jinnai", s3_path)
            continue

        if prompt == "None":
            input_source = src_lines[sample_id]
            model_inputs = tokenizer(
                input_source, return_tensors="pt", truncation=True
            ).to(torch_device)
            stopping_criteria = None
        else:
            # TODO: Refactor the prompt handling
            if "zephyr" in model_n:
                # Zero shot prompting.
                # TODO: Implement few shot prompting.
                messages = [
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": src_lines[sample_id],
                    },
                ]
                input_source = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            elif "pythia" in model_n:
                input_source = tokenizer.apply_chat_template(
                    src_lines[sample_id], tokenize=False, add_generation_prompt=True
                )
            elif "dolly-v2-3b" in model_n:
                # TODO: this only works for alpaca format.
                INSTRUCTION_KEY = "### Instruction:"
                RESPONSE_KEY = "### Response:"
                END_KEY = "### End"
                INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                input_source = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
                    intro=INTRO_BLURB,
                    instruction_key=INSTRUCTION_KEY,
                    instruction=src_lines[sample_id][0]["content"],
                    response_key=RESPONSE_KEY,
                )
                if sample_id == 0:
                    print("input_source=", input_source)
            elif ("japanese-large-lm-3.6b-instruction-sft" in model_n) or (
                "dpo_20240122-153719" in model_n
            ):
                prompt = "ユーザー: {}\nシステム: "
                input_source = prompt.format(src_lines[sample_id])
            # elif 'llm-jp' in model_n:
            #     prompt = "{}### 回答："
            #     input_source = prompt.format(src_lines[sample_id])
            elif "TowerInstruct" in model_n:
                messages = [
                    {
                        "role": "user",
                        "content": "Translate the following text from German into English.\nGerman: {}\nEnglish:".format(
                            src_lines[sample_id]
                        ),
                    },
                ]
                input_source = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            elif "42dot" in model_n:
                prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request in English."
                messages = [
                    {
                        "role": "user",
                        "content": prompt + "\n\n" + src_lines[sample_id][0]["content"],
                    }
                ]
                input_source = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            elif "paligemma" in model_n:
                input_source = prompt
            elif "[[QUESTION]]" not in prompt:
                if isinstance(src_lines[sample_id], list):
                    input_source = tokenizer.apply_chat_template(
                        src_lines[sample_id], tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt = """USER: {}
ASSISTANT: """
                    input_source = prompt.format(src_lines[sample_id])
                if sample_id == 0:
                    print("input_source=", input_source)
            else:
                input_source = prompt.replace("[[QUESTION]]", src_lines[sample_id])
                if "Mistral" in model_n:
                    input_source = "[INST] " + input_source + "[/INST]"
                else:
                    messages = [
                        {"role": "user", "content": input_source},
                    ]
                    input_source = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

            if sample_id == 0:
                print("input_source=", input_source)

            # if ('japanese-large-lm-' in model_n) or ('llm-jp' in model_n) or
            # ('llmjp' in model_n) or ('dpo_20240122-153719' in model_n) or
            # ('line-' in model_n):
            if (
                ("japanese-large-lm-" in model_n)
                or ("dpo_20240122-153719" in model_n)
                or ("line-" in model_n)
            ):
                # TODO: LINE's tokenizer put EOS at the end of the sentence.
                # This is a hack to remove it.
                model_inputs = tokenizer(
                    input_source,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    add_special_tokens=False,
                ).to(model.device)
            elif "paligemma" in model_n:
                model_inputs = tokenizer(
                    prompt, src_lines[sample_id], return_tensors="pt"
                ).to(model.device)
            else:
                model_inputs = tokenizer(
                    input_source, return_tensors="pt", return_token_type_ids=False
                ).to(model.device)
            input_length = model_inputs["input_ids"].shape[1]

            # This makes the model not to generate more than a line of
            # sentence.

            if len(stop_tokens) > 0:
                bins = torch.bincount(model_inputs["input_ids"][0].to("cpu"))
                nlines = bins[stop_tokens[0]].numpy()
                stopping_criteria = StoppingCriteriaList(
                    [StoppingCriteriaSub(stops=stop_tokens, encounters=nlines + 1)]
                )
            else:
                # TODO: do not stop. This is not an efficient implementation,
                # used only for debugging purpose.
                stopping_criteria = None

        set_seed(42)

        rows = []

        for i in range(n_batches):
            num_return_sequences = bsz
            if do_sample > 0:
                sample_output = model.generate(
                    **model_inputs,
                    **model_kwargs,
                    do_sample=True,
                    temperature=temperature,
                    epsilon_cutoff=eps,
                    top_k=topk,
                    top_p=topp,
                    num_beams=1,
                    num_return_sequences=num_return_sequences,
                    stopping_criteria=stopping_criteria,
                    return_dict_in_generate=True,
                    output_scores=True,
                    forced_bos_token_id=model.config.forced_bos_token_id,
                )
            else:
                if diversity_penalty > 0.000001:
                    num_beam_groups = bsz
                else:
                    # If diversity penalty is zero, then DiverseBS is
                    # equivalent to beam search.
                    num_beam_groups = 1

                if do_sample < 0:
                    # Return only one sequence for beam search.
                    num_return_sequences = 1
                    # computation of the probability is not implemented for
                    # beam search for now.
                    discard_prob = True
                else:
                    # Return bsz sequences for diverse beam search.
                    num_return_sequences = bsz

                sample_output = model.generate(
                    **model_inputs,
                    **model_kwargs,
                    do_sample=False,
                    temperature=temperature,
                    epsilon_cutoff=eps,
                    top_k=topk,
                    top_p=topp,
                    num_beams=bsz,
                    num_return_sequences=num_return_sequences,
                    num_beam_groups=num_beam_groups,
                    diversity_penalty=diversity_penalty,
                    stopping_criteria=stopping_criteria,
                    return_dict_in_generate=True,
                    output_scores=True,
                    forced_bos_token_id=model.config.forced_bos_token_id,
                )

            if prompt == "None":
                if discard_prob:
                    probs = None
                elif do_sample > 0:
                    probs = compute_probability_s2s(sample_output)
                else:
                    # Ignore the probability for beam search.
                    # The output shape is different from sampling which I would
                    # expect to be fixed in future.
                    probs = [0.0] * num_return_sequences
                for j in range(num_return_sequences):
                    sample_text = tokenizer.decode(
                        sample_output.sequences[j], skip_special_tokens=True
                    )
                    rows.append((sample_text, probs[j]))
            else:
                if discard_prob:
                    output_prob = [None] * num_return_sequences
                else:
                    output_prob = compute_probability_lm(model, sample_output)
                output_text = get_texts(tokenizer, sample_output, input_length)
                for j in range(num_return_sequences):
                    rows.append((output_text[j], output_prob[j]))

        if temperature != 1.0:
            filename = "{:04}_eps-{:.2f}_topk-{:02d}_topp-{:.2f}_tmp-{:.2f}".format(
                sample_id, eps, topk, topp, temperature
            )
        elif do_sample > 0:
            filename = "{:04}_eps-{:.2f}_topk-{:02d}_topp-{:.2f}".format(
                sample_id, eps, topk, topp
            )
        elif do_sample == 0:
            filename = "{:04}_beam-{:02d}_divpen-{:.2f}".format(
                sample_id, bsz, diversity_penalty
            )
        elif do_sample < 0:
            filename = "{:04}_beam-{:02d}".format(sample_id, bsz)

        outfilepath = os.path.join(sample_dir, dataset, model_n, filename)

        df = pd.DataFrame(rows, columns=["text", "probability"])
        # df.to_csv(outfilepath, index=False)
        # This is to prevent the double backslash issue.
        # https://stackoverflow.com/questions/61924003/how-to-escape-the-escapechar-in-pandas-to-csv

        # df.to_csv(outfilepath, index=False, float_format='%.32f')
        df.to_csv(
            outfilepath,
            sep=",",
            escapechar="\\",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            float_format="%.32f",
        )
        s3_path = os.path.join("fairseq", "mbr", "samples", dataset, model_n, filename)
        client.upload_file(outfilepath, "ailab-jinnai", s3_path)


if __name__ == "__main__":
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    # To prevent "cutlassF: no kernel found to launch!" erorr
    # torch.backends.cuda.enable_mem_efficient_sdp(False)
    # torch.backends.cuda.enable_flash_sdp(False)

    parser = get_mbr_parser()
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model
    prompt_path = args.prompt
    n_lines = args.n_lines
    start_iter = args.start_iter

    n_samples = args.n_samples
    bsz = args.bsz
    temperature = args.temperature
    eps = args.eps
    topk = args.topk
    topp = args.topp
    do_sample = args.do_sample
    diversity_penalty = args.diversity_penalty

    max_new_tokens = args.max_new_tokens

    quantize = args.quantize

    discard_prob = args.discard_prob

    src_lines = load_dataset(dataset)
    tokenizer, model, model_name, stop_tokens = load_model(
        dataset, torch_device, model_name, quantize
    )

    if prompt_path == "None":
        prompt = "None"
    else:
        with open(os.path.join(prompt_dir, prompt_path), "r") as f:
            prompt = f.read()

    sample(
        dataset,
        tokenizer,
        model,
        src_lines,
        torch_device,
        n_lines,
        start_iter,
        n_samples,
        bsz,
        temperature,
        eps,
        topk,
        topp,
        do_sample,
        diversity_penalty,
        prompt,
        stop_tokens,
        model_n=os.path.basename(model_name),
        discard_prob=discard_prob,
        max_new_tokens=max_new_tokens,
    )
