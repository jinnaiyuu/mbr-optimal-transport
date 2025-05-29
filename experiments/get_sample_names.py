
def get_sample_file_pattern(epsilon, topk, topp, do_sample, diverse_k, divpen, temperature=1.0):
    """
    This function returns the file pattern for the sample files.
    Using this function, we can download only the files we need.
    Otherwise, we have to download all the files in the directory which amount to >10,000.
    """
    if do_sample > 0:
        if topk < 0 and topp < 0:
            return "*_eps-{:.2f}".format(epsilon)
        elif temperature == 1.0:
            return "*_eps-{:.2f}_topk-{:02d}_topp-{:.2f}".format(epsilon, topk, topp)
        else:
            return "*_eps-{:.2f}_topk-{:02d}_topp-{:.2f}_tmp-{:.2f}".format(epsilon, topk, topp, temperature)
    elif do_sample == 0:
        return "*_beam-{:02d}_divpen-{:.2f}".format(diverse_k, divpen)
    else:
        return "*_beam-{:02d}".format(diverse_k)


if __name__ == "__main__":
    import sys
    epsilon = float(sys.argv[1])
    topk = int(sys.argv[2])
    topp = float(sys.argv[3])
    do_sample = int(sys.argv[4])
    diverse_k = int(sys.argv[5])
    divpen = float(sys.argv[6])

    if len(sys.argv) > 7:
        temperature = float(sys.argv[7])
    else:
        temperature = 1.0
    print(get_sample_file_pattern(epsilon, topk, topp, do_sample, diverse_k, divpen, temperature))
