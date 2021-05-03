import json
from table import *
from table_ast import *
from synthesizer import *
from tabulate import tabulate
from table_cell import *
from configuration import target_configs
import logging
import os
import time

pd.set_option('expand_frame_repr', False)

DATA_DIR = os.path.join(".", "testbenches")

# Creating an object
# logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
# logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s %(message)s')

logging.basicConfig(format='%(asctime)s %(message)s',
                    filemode='w+',
                    level=logging.INFO)
random.seed(5)


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, mode='w+')
    # handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


logger_summary = setup_logger("summary", f'../eval/summary1.log')


def permutate_table(annotated_output):
    columns = [i for i in range(annotated_output.get_col_num())]
    permutation_list = list(itertools.permutations(columns, annotated_output.get_col_num()))
    # logger.info(permutation_list)  # verify permutations of column ids
    return [select_columns(annotated_output, selected)
            for selected in permutation_list]


def randomize_table(annotated_output, config, logger, permutation_candidates):
    if config["permutation_test"]:
        # if config["random_test"]:
        sample_id = random.randrange(len(permutation_candidates))
        annotated_output = permutation_candidates[sample_id]
        logger.info("=======output candidates " + str(sample_id) + "==========")
        logger.info(annotated_output.to_dataframe())
        logger.info("===============================")
    if config["partial_table"]:
        if config["random_test"]:
            x_start = random.randrange(int(annotated_output.get_col_num() / 2))
            y_start = random.randrange(int(annotated_output.get_row_num() / 2))
            x_end = random.randrange(int(annotated_output.get_col_num() / 2), annotated_output.get_col_num())
            y_end = random.randrange(int(annotated_output.get_row_num() / 2), annotated_output.get_row_num())
        else:
            x_end = annotated_output.get_col_num()
            x_start = int(annotated_output.get_col_num() / 2)
            y_end = annotated_output.get_row_num()
            y_start = int(annotated_output.get_row_num() / 2)
        annotated_output = annotated_output.select_region((x_start, x_end), (y_start, y_end))
        logger.info("=======with partial table==========")
        logger.info(annotated_output.to_dataframe())

    if config["partial_trace"]:
        annotated_output = annotated_output.randomize()
        logger.info("=======with randomized trace==========")
        logger.info(annotated_output.to_dataframe())


def eval_correctness(inputs, candidates, correct_out):
    results = []
    for p in candidates:
        if checker_function(p.eval(inputs), correct_out) is not None:
            results.append(p)
    return results


def run_wrapper(annotated_output, correct_out, config, logger):
    # logger.info(p.eval(inputs).to_dataframe())
    # logger.error(f"[error] invalid benchmark file")
    candidates = []
    results = []
    for j in range(config["level_limit"], config["level_limit"] + 1):
        candidates = Synthesizer(config).run_synthesis(inputs, annotated_output, j, logger, correct_out,
                                                        with_analysis=config["with_analysis"],
                                                        solution_limit=config["solution_limit"],
                                                        time_limit_sec=config["time_limit"],
                                                        print_trace=False)
        # results = eval_correctness(inputs, candidates, correct_out)
        #if len(candidates) > 0:
        #
        break
    # eval on the correct output
    logger_summary.info(f"number of correct solutions: {len(candidates)}")

    logger.info("=======target output==========")
    logger.info(annotated_output.to_dataframe())
    for p in candidates:
        # print(alignment_result)
        logger.info(p.stmt_string())
        logger.info(tabulate(p.eval(inputs).extract_values(), headers='keys', tablefmt='psql'))
        logger.info(tabulate(p.eval(inputs).extract_traces(), headers='keys', tablefmt='psql'))
        logger.info("\n")
    logger.info(f"number of programs: {len(candidates)}")
    logger.info("\n\n\n\n\n\n")
    logger.info("------------------------------------------------------------------------------------------")


if __name__ == '__main__':
    config = {
        "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
        "filer_op": ["=="],
        "constants": [3000],
        "aggr_func": ["mean", "sum", "count", "max"],
        "mutate_func": ["mean", "sum", "max", "count", "cumsum"],
        "join_predicates": ["(0, 0)", "(1, 0)", "(0, 1)"],
        "mutate_function": ["lambda x, y: x - y",
                            "lambda x, y: x + y",
                            "lambda x, y: x * y",
                            "lambda x, y: x / y",
                            "lambda x: x - (x * 0.1)",
                            "lambda x, y: y / (x - y)",
                            "lambda x: 1"],
        "permutation_test": False,
        "random_test": False,
        "partial_table": False,
        "partial_trace": False,
        "level_limit": 4,
        "time_limit": 600,
        "solution_limit": 5
    }

    for fname in os.listdir(DATA_DIR):
    # for fname in ["006.json", "001.json"]:
        if fname == "024.json":  # add more arithmetic operators later
            continue
        if fname == "016.json":  # table too big
            continue
        if fname.endswith("json") and "discard" not in fname:
            fpath = os.path.join(DATA_DIR, fname)
            with open(fpath, 'r') as filehandler:
                # file info
                logger_summary.info(f"<==============evaluation of {fname}================>")
                data = json.load(filehandler)
                # description:
                inputs = data["input_data"]
                # log inputs
                input_info = []
                for inp in inputs:
                    df = pd.DataFrame.from_dict(inp)
                    input_info.append(f"{len(df.columns)}x{len(df)}")
                logger_summary.info(f"---input info:{input_info}")

                # get the labelled correct program
                correct_p = dict_to_program(data["exp_out"])
                correct_out = correct_p.eval(inputs)
                logger_summary.info(f"target program:{correct_p.stmt_string()}")
                permutation_candidates = permutate_table(correct_out)
                # run on all configs
                user_example = copy.copy(correct_out)
                for i in range(len(target_configs)):
                    curr_config = target_configs[i]
                    logger = setup_logger(f"{fname}_{i}", f'../eval/{fname[:3]}_config({i}).log')
                    config = target_configs[i]
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
                    if i % 2 == 0:
                        randomize_table(user_example, config, logger, permutation_candidates)
                    try:
                        run_wrapper(user_example, correct_out, config, logger)
                    except Exception as e:
                        logger_summary.info(f"[Error] examining {fname}")
                        print(f"[error] {sys.exc_info()[0]} {e}")
                        tb = sys.exc_info()[2]
                        tb_info = ''.join(traceback.format_tb(tb))
                        print(tb_info)
                        continue
                    print("<=====Finish")
                logger_summary.info("\n")
    print("<=====Evaluation Ends")
