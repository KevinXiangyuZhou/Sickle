import json
from table import *
from table_ast import *
from synthesizer import *
from tabulate import tabulate
from table_cell import *
from configuration import target_configs
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('expand_frame_repr', False)

DATA_DIR = os.path.join(".", "testbenches")
# DATA_DIR = os.path.join(".", "../benchmark/tpc-ds")
# storage of current data
run_time_summary = []
n_program_search_summary = []
run_time_summary_analysis = []
n_program_search_summary_analysis = []
run_time_result = {}
n_program_result = {}
# Creating an object
# logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
# logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s %(message)s')

logging.basicConfig(format='%(asctime)s %(message)s',
                    filemode='w+',
                    level=logging.INFO)
random.seed(123)


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, mode='w+')
    # handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


logger_summary = setup_logger("summary", f'../eval/summary.log')
logger_data = setup_logger("data", f'../eval/data_8.16.log')


def barchart(data_1, data_2, x_label, y_label, fnames, img_name):
    # plt.yscale('log')
    ind = np.arange(len(fnames))
    width = 0.35
    plt.bar(fnames, data_1, width, label="default")
    plt.bar(ind + width, data_2, width,
        label='with_analysis')

    plt.ylabel(y_label)
    plt.title(f"{x_label} vs {y_label}")

    plt.xlabel(x_label)
    plt.xticks(ind + width / 2, fnames)
    plt.legend(["enumeration", "enumeration with analysis"])
    plt.savefig(f'../plots/{img_name}.png')
    plt.show()


def permutate_table(annotated_output):
    columns = [i for i in range(annotated_output.get_col_num())]
    permutation_list = list(itertools.permutations(columns, annotated_output.get_col_num()))
    # logger.info(permutation_list)  # verify permutations of column ids
    if len(permutation_list) > 10:
        permutation_list = permutation_list[:10]
    return [select_columns(annotated_output, selected)
            for selected in permutation_list]


def randomize_table(annotated_output, config, logger):
    res = copy.copy(annotated_output)
    logger.info("=======output candidate ==========")
    logger.info(res.to_dataframe())
    logger.info("===============================")
    if config["partial_table"]:
        """
        x_s, x_e = int(res.get_col_num() / 2), res.get_col_num()
        y_s, y_e = int(res.get_row_num() / 2), res.get_row_num()
        if x_s == 0:
            x_start = random.randrange(x_s + 1)
        else:
            x_start = random.randrange(x_s)
        if y_s == 0:
            y_start = random.randrange(y_s + 1)
        else:
            y_start = random.randrange(y_s)
        # x_end = random.randrange(x_s, x_e)
        if config["random_test"]:
            # x_start = 0
            # y_start = 0
            # x_end = res.get_col_num()
            x_end = random.randrange(x_s, x_e)
            if y_e < 2:
                y_end = res.get_row_num()
            else:
                y_end = 2
            y_end = random.randrange(y_s, y_e)
        else:
            x_start, y_start = 0, 0
            x_end, y_end = res.get_col_num(), 2
        """
        if res.get_row_num() < 2:
            selected_rids = [0]
        else:
            selected_rids = random.sample(range(res.get_row_num()), 2)
    else:
        """
        x_end = res.get_col_num()
        x_start = int(res.get_col_num() / 2)
        y_end = res.get_row_num()
        y_start = int(res.get_row_num() / 2)
        """
        selected_rids = [i for i in range(res.get_row_num())]
    # res = res.select_region((x_start, x_end), (y_start, y_end))
    res = res.select_rows(selected_rids)
    logger.info("=======with partial table==========")
    logger.info(res.to_dataframe())

    if config["partial_trace"]:
        res = res.randomize()
        logger.info("=======with randomized trace==========")
        logger.info(res.to_dataframe())
    if config["permutation_test"]:
        permutation_candidates = permutate_table(res)
        sample_id = random.randrange(len(permutation_candidates))
        res = permutation_candidates[sample_id]
        logger.info("=======permutated output:" + str(sample_id) + "==========")
        logger.info(res.to_dataframe())
    return res


def eval_correctness(inputs, candidates, correct_out):
    results = []
    for p in candidates:
        if checker_function(p.eval(inputs), correct_out) is not None:
            results.append(p)
    return results


def run_wrapper(inputs, annotated_output, correct_out, config, logger):
    # logger.info(p.eval(inputs).to_dataframe())
    # logger.error(f"[error] invalid benchmark file")
    candidates = []
    for j in range(config["level_limit"], config["level_limit"] + 1):
        candidates, run_time, n_program = Synthesizer(config).run_synthesis(inputs, annotated_output, j, logger,
                                                                            correct_out,
                                                                            with_analysis=config["with_analysis"],
                                                                            use_val=config["value_analysis"],
                                                                            solution_limit=config["solution_limit"],
                                                                            time_limit_sec=config["time_limit"],
                                                                            print_trace=False)
        # results = eval_correctness(inputs, candidates, correct_out)
        break
    # eval on the correct output
    logger_summary.info(f"number of correct solutions: {len(candidates)}")

    logger.info("=======target output==========")
    logger.info(annotated_output.to_dataframe())
    logger.info("=======correct output==========")
    logger.info(correct_out.to_dataframe())
    for p in candidates:
        # print(alignment_result)
        res = p.eval(inputs)
        logger.info(p.stmt_string())
        logger.info(tabulate(res.extract_values(), headers='keys', tablefmt='psql'))
        logger.info(tabulate(res.extract_traces(), headers='keys', tablefmt='psql'))
        logger.info("\n")
    logger.info(f"number of programs: {len(candidates)}")
    logger.info("\n\n\n\n\n\n")
    logger.info("------------------------------------------------------------------------------------------")
    return run_time, n_program, len(candidates)


if __name__ == '__main__':
    fnames = []
    logger_data.info("{\"data\":[")
    for fname in os.listdir(DATA_DIR):
    # for fname in ["009.json", "014.json", "018.json", "021.json", "029.json", "034.json", "036.json"]:
    # for fname in ["005.json", "009.json", "111.json"]:
    # for fname in ["113.json", "107.json", "120.json"]:
        # tpc-ds
        if fname.endswith("json") and "discard" not in fname:
            fnames.append(fname[:3])
            fpath = os.path.join(DATA_DIR, fname)
            with open(fpath, 'r') as filehandler:
                run_time_result[fname] = {}
                n_program_result[fname] = {}
                # file info
                logger_summary.info(f"<==============evaluation of {fname}================>")
                data = json.load(filehandler)
                # description:
                inputs = data["input_data"]
                # log inputs
                input_info = []
                cell_count = 0
                for inp in inputs:
                    df = pd.DataFrame.from_dict(inp)
                    input_info.append(f"{len(df.columns)}x{len(df)}")
                    cell_count += len(df.columns) * len(df)
                logger_summary.info(f"---input info:{input_info}")

                # get the labelled correct program
                correct_p = dict_to_program(data["exp_out"])
                logger_summary.info(f"target program:{correct_p.stmt_string()}")
                # permutation_candidates = permutate_table(correct_out)
                # run on all configs
                for i in range(len(target_configs)):
                    # for j in range(5):
                    curr_config = target_configs[i]
                    if curr_config["row_limit"] is None:
                        config_inputs = inputs
                    else:
                        config_inputs = [data[:curr_config["row_limit"]] for data in inputs]
                    # print(config_inputs)
                    input_info = []
                    for inp in config_inputs:
                        df = pd.DataFrame.from_dict(inp)
                        input_info.append(f"{len(df.columns)}x{len(df)}")
                    correct_out = correct_p.eval(config_inputs)
                    # if i == 1 or i == 3:
                    #     continue
                    # if DATA_DIR == os.path.join(".", "../benchmark/tpc-ds") and i == 0:
                    #     continue
                    # if i < 4:  # select configs
                    #     continue
                    user_example = copy.copy(correct_out)
                    if "parameter_config" in data.keys():
                        curr_config["parameter_config"] = data["parameter_config"]
                    logger = setup_logger(f"{fname}_{i}", f'../eval/{fname[:3]}_config({i}).log')
                    # for config with analysis, run ten times
                    print("START=====>")
                    print(f"evaluate {fname} on config_{i}...")
                    logger_summary.info(f"------config {i}-------")
                    # log config info
                    logger_summary.info(json.dumps({key: curr_config[key] for key in curr_config
                                                    if key == "with_analysis"
                                                    or key == "permutation_test"
                                                    or key == "partial_table"
                                                    or key == "partial_trace"
                                                    or key == "level_limit"
                                                    or key == "time_limit"
                                                    or key == "solution_limit"}, indent=3))
                    logger.info(f"------evaluate {fname} on config_{i}-------")
                    logger.info(str(curr_config))
                    # get randomized user sample, according to the given config
                    print("=======correct p==========")
                    print(correct_p.stmt_string())
                    user_example = user_example.compress_sum()
                    user_example = randomize_table(user_example, curr_config, logger)
                    try:
                        run_time_j, n_program_j, num_candidates = run_wrapper(config_inputs, user_example, correct_out, curr_config, logger)
                        # run_time_j, n_program_j, num_candidates = 0, 0, []
                    except Exception as e:
                        logger_summary.info(f"[Error] examining {fname}")
                        print(f"[error] {sys.exc_info()[0]} {e}")
                        tb = sys.exc_info()[2]
                        tb_info = ''.join(traceback.format_tb(tb))
                        print(tb_info)
                        continue

                    # log result for current config
                    log_data = {}
                    log_data["id"] = int(fname[:3])
                    log_data["num_program"] = len(data["exp_out"]) - 1
                    log_data["input_size"] = input_info
                    log_data["user_example_size"] = \
                        f"{user_example.get_row_num()}x{user_example.get_col_num()}"
                    log_data["input_cells"] = cell_count
                    log_data["output_cells"] = correct_out.get_row_num() * correct_out.get_col_num()
                    log_data["user_example_cells"] = user_example.get_row_num() * user_example.get_col_num()
                    if curr_config["with_analysis"] and not curr_config["value_analysis"]:
                        log_data["user_ptrs"] = user_example.count_ptrs()
                        log_data["output_ptrs"] = correct_out.count_ptrs(count_or=True)
                    else:
                        log_data["user_ptrs"] = 0
                        log_data["output_ptrs"] = 0
                    if curr_config["with_analysis"]:
                        log_data["with_pruning_or_not"] = 1
                    else:
                        log_data["with_pruning_or_not"] = 0
                    if curr_config["value_analysis"]:
                        log_data["analysis_type"] = "value"
                    else:
                        log_data["analysis_type"] = "trace"
                    log_data["time"] = run_time_j
                    log_data["num_program_visited"] = n_program_j
                    log_data["num_consistent"] = num_candidates
                    # log_data["size"] = curr_config["row_limit"]
                    if run_time_j >= curr_config["time_limit"]:
                        log_data["timeout"] = 1
                    else:
                        log_data["timeout"] = 0
                    if fname == os.listdir(DATA_DIR)[-1] and i == len(target_configs) - 1:
                        logger_data.info(f"{str(log_data)}")
                    else:
                        logger_data.info(f"{str(log_data)},")

                    print("<=====Finish")

    logger_data.info("]}")
    print("<=====Evaluation Ends")

