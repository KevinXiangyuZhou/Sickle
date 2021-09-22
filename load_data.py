import numpy as np
import os
import csv
import pandas as pd
import math
import logging


logging.basicConfig(format='%(asctime)s %(message)s',
                    filemode='w+',
                    level=logging.INFO)

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, mode='w+')
    # handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

np.random.seed(123)

dataset_07 = {
    "store_sales.dat": {2: "ss_item_sk", 10: "ss_quantity", 12: "ss_list_price", 19: "ss_coupon_amt",
                        13: "ss_sales_price"},
    "item.dat": {0: "i_item_sk", 1: "i_item_id"}
}
dataset_20 = {
    "catalog_sales.dat": {15: "cs_item_sk", 0: "cs_sold_date_sk", 23: "cs_ext_sales_price"},
    "item.dat": {0: "i_item_sk", 1: "i_item_id", 4: "i_item_desc", 12: "i_category", 10: "i_class",
                 5: "i_current_price"},
    "date_dim.dat": {0: "d_date_sk", 2: "d_date"}
}

dataset_23 = {
    "store_sales.dat": {3: "ss_customer_sk", 0: "ss_sold_date_sk", 10: "ss_quantity", 13: "ss_sales_price"},
    "customer.dat": {0: "c_customer_sk"},
    "date_dim.dat": {0: "d_date_sk", 2: "d_date"}
}
# not usable
dataset_31a = {
    "store_sales.dat": {3: "ss_customer_sk", 10: "ss_quantity", 13: "ss_ext_sales_price"},
    "customer_address.dat": {0: "ca_address_sk", 1: "ca_county"},
    "date_dim.dat": {0: "d_date_sk", 6: "d_year"}
}

dataset_36 = {
    "store_sales.dat": {2: "ss_item_sk", 0: "ss_sold_date_sk", 22: "ss_net_profit", 15: "ss_ext_sales_price"},
    "date_dim.dat": {0: "d_date_sk", 6: "d_year"},
    "item.dat": {0: "i_item_sk", 12: "i_category", 10: "i_class"},
}

dataset_47 = {
    "store_sales.dat": {2: "ss_item_sk", 0: "ss_sold_date_sk", 13: "ss_sales_price"},
    "date_dim.dat": {0: "d_date_sk", 6: "d_year"},
    "item.dat": {0: "i_item_sk", 12: "i_category", 8: "i_brand"}
}

dataset_58 = {
    "store_sales.dat": {2: "ss_item_sk", 15: "ss_ext_sales_price"},
    "catalog_sales.dat": {15: "cs_item_sk", 23: "cs_ext_sales_price"},
    "item.dat": {0: "i_item_sk", 1: "i_item_id"}
}

dataset_72 = {
    "promotion.dat": {0: "p_promo_sk"},
    "catalog_sales.dat": {15: "cs_item_sk", 16: "cs_promo_sk"},
    "item.dat": {0: "i_item_sk", 1: "i_item_id"}
}

dataset_80 = {
    "store_sales.dat": {0: "ss_item_sk", 22: "ss_net_profit"},
    "store_returns.dat": {2: "sr_item_sk", 1:"sr_net_loss", 11:"sr_return_amt"}
}

dataset_81 = {
    "catalog_returns.dat": {0: "cr_returned_date_sk", 10: "cr_returning_addr_sk",
                            20:"cr_return_amt_inc_tax", 3:"cr_returning_customer_sk"},
    "date_dim.dat": {0: "d_date_sk", 2: "d_date"},
    "customer_address.dat": {0: "ca_address_sk"}
}
"""
dataset_83 = {
    "catalog_returns.dat": {0: "cr_returned_date_sk", 1: "cr_returning_addr_sk"},
    "date_dim.dat": {15: "d_date_sk"},
    "customer_address.dat": {15: "ca_address_sk"}
}
"""

dataset_85 = {
    "web_sales.dat": {0: "ws_sold_date_sk", 3: "ws_item_sk", 18: "ws_quantity"},
    "web_returns.dat": {2: "wr_item_sk", 20: "wr_refunded_cash", 18:"wr_fee"},
    "date_dim.dat": {0: "d_date_sk", 2: "d_date"}
}

dataset_86 = {
    "web_sales.dat": {0: "ws_sold_date_sk", 3: "ws_item_sk", 29:"ws_net_paid"},
    "item.dat": {0: "i_item_sk", 12: "i_category", 10: "i_class"},
    "date_dim.dat": {0: "d_date_sk", 2: "d_date"}
}

dataset_89 = {
    "store_sales.dat": {2: "ss_item_sk", 0: "ss_sold_date_sk", 13: "ss_sales_price"},
    "date_dim.dat": {0: "d_date_sk", 8: "d_moy"},
    "item.dat": {0: "i_item_sk", 8: "i_brand", 12: "i_category", 10: "i_class"},
}

dataset_92 = {
    "web_sales.dat": {2: "ws_item_sk", 0: "ws_sold_date_sk", 22: "ws_ext_discount_amt"},
    "date_dim.dat": {0: "d_date_sk", 2: "d_date"},
    "item.dat": {0: "i_item_sk"}
}

dataset_93 = {
    "store_sales.dat": {0: "ss_item_sk", 10: "ss_quantity", 3: "ss_customer_sk", 13: "ss_sales_price"},
    "store_returns.dat": {2: "sr_item_sk", 10: "sr_return_quantity"}
}

dataset_94 = {
    "web_sales.dat": {2: "ws_ship_date_sk",
                      11: "ws_ship_addr_sk", 28: "ws_ext_ship_cost", 33: "ws_net_profit"},
    "date_dim.dat": {0: "d_date_sk", 2: "d_date"},
    "customer_address.dat": {0: "ca_address_sk"}
}

dataset_98 = {
    "store_sales.dat": {2: "ss_item_sk", 0: "ss_sold_date_sk", 15: "ss_ext_sales_price"},
    "item.dat": {0: "i_item_sk", 1: "i_item_id", 4: "i_item_desc", 12: "i_category", 10: "i_class",
                 5: "i_current_price"},
    "date_dim.dat": {0: "d_date_sk", 2: "d_date"}
}

dataset_63 = {
    "store_sales.dat": {2: "ss_item_sk", 0: "ss_sold_date_sk", 13: "ss_sales_price"},
    "item.dat": {0: "i_item_sk", 20: "i_manager_id"},
    "date_dim.dat": {0: "d_date_sk", 8: "d_moy"}
}

dataset_64 = {
    "catalog_sales.dat": {15: "cs_item_sk", 25:"cs_ext_list_price"},
    "catalog_returns.dat": {2: "cr_item_sk", 23: "cr_refunded_cash", 24:"cr_reversed_charge",
                            25: "cr_store_credit"}
}

sk_ids_07 = {"item.dat": [0], "store_sales.dat": [2]}
sk_ids_20 = {"item.dat": [0], "date_dim.dat": [0], "catalog_sales.dat": [15, 0]}
sk_ids_23 = {"customer.dat": [0], "date_dim.dat": [0], "store_sales.dat": [3, 0]}
sk_ids_36 = {"item.dat": [0], "date_dim.dat": [0], "store_sales.dat": [2, 0]}
sk_ids_47 = {"item.dat": [0], "date_dim.dat": [0], "store_sales.dat": [2, 0]}
sk_ids_58 = {"item.dat": [0], "store_sales.dat": [2], "catalog_sales.dat": [15]}
sk_ids_72 = {"item.dat": [0], "promotion.dat": [0], "catalog_sales.dat": [15, 16]}
sk_ids_80 = {"store_sales.dat": [0], "store_returns.dat": [2]}
sk_ids_81 = {"customer_address.dat": [0], "date_dim.dat": [0], "catalog_returns.dat": [0, 10]}
sk_ids_85 = {"date_dim.dat": [0], "web_returns.dat": [2], "web_sales.dat": [0, 3]}
sk_ids_86 = {"item.dat": [0], "date_dim.dat": [0], "web_sales.dat": [0, 3]}
sk_ids_89 = {"item.dat": [0], "date_dim.dat": [0], "store_sales.dat": [2, 0]}
sk_ids_92 = {"item.dat": [0], "date_dim.dat": [0], "web_sales.dat": [2, 0]}
sk_ids_93 = {"store_sales.dat": [0], "store_returns.dat": [2]}
sk_ids_94 = {"customer_address.dat": [0], "date_dim.dat": [0], "web_sales.dat": [2, 11]}
sk_ids_98 = {"item.dat": [0], "date_dim.dat": [0], "store_sales.dat": [2, 0]}
sk_ids_63 = {"item.dat": [0], "date_dim.dat": [0], "store_sales.dat": [2, 0]}
sk_ids_64 = {"catalog_sales.dat": [15], "catalog_returns.dat": [2]}

data_020 = {0: "ss_asdas"}

data_loader = {7: [dataset_07, sk_ids_07],
               20: [dataset_20, sk_ids_20],
               23: [dataset_23, sk_ids_23],
               36: [dataset_36, sk_ids_36],
               47: [dataset_47, sk_ids_47],
               58: [dataset_58, sk_ids_58],
               72: [dataset_72, sk_ids_72],
               80: [dataset_80, sk_ids_80],
               81: [dataset_81, sk_ids_81],
               85: [dataset_85, sk_ids_85],
               86: [dataset_86, sk_ids_86],
               89: [dataset_89, sk_ids_89],
               92: [dataset_92, sk_ids_92],
               93: [dataset_93, sk_ids_93],
               94: [dataset_94, sk_ids_94],
               98: [dataset_98, sk_ids_98],
               63: [dataset_63, sk_ids_63],
               64: [dataset_64, sk_ids_64]}


def csv_parser(filename, sk_ids):
    path = os.path.join('..\TPC-DS benchmarks\data', filename)
    datafile = open(path, 'r')
    datareader = csv.reader(datafile, delimiter='|')
    # reviews_df = pd.read_csv(path, nrows=100)
    data = []
    # the column length should be determined by the columns of the first row
    first = next(datareader)
    data.append(process_row(first))
    colnum = len(first)
    sk_set = {cid: set([]) for cid in sk_ids[filename]}
    for row in datareader:
        if len(row) != colnum:
            continue
        # change value type
        processed = process_row(row)
        if not valid(processed[0:len(processed)-1]):
            continue
        for cid in range(len(processed)):
            if cid in sk_set.keys():
                val = processed[cid]
                sk_set[cid].add(val)
                processed[cid] = len(sk_set[cid])
        data.append(processed)
    sk_range = {cid: (data[0][cid], data[-1][cid]) for cid in sk_ids[filename]}
    print(sk_range)
    return data, sk_range

def valid(row):
    for val in row:
        if val == '':
            return False
    return True

def process_row(row):
    processed = []
    for val in row:
        try:
            processed.append(float(val))
        except:
            processed.append(val)
    return processed


# filter out the rows that contain the same join
def filter_range(skr1, skr2):
    return [max(skr1[0], skr2[0]), min(skr1[1], skr2[1])]


def build_json(data, cols, size=10):
    # indexes = np.random.choice(np.arange(len(data)), size)
    json_data = []
    # if row_limit is not None and len(data) > row_limit:
    #    data = data[:row_limit]
    rid = 0
    for row in data:
        new_row = {}
        # if rid not in indexes:
        #     rid += 1
        #     continue
        for col in cols.keys():
            new_row[cols[col]] = row[col]
        # print(str(new_row) + ",")
        json_data.append(new_row)
        rid += 1
    return json_data


def filter_table(data_dict, values, cid):
    new_source = []
    for row in data_dict:
        # print(row)
        if row[cid] == "":
            continue
        if row[cid] in values:
            new_source.append(row)
    return new_source


def load_data(data_id):
    data_load_app = data_loader[data_id]
    file_list = list(data_load_app[0].keys())
    filter_range = np.arange(2450914, 2450926)
    sk_col = 0
    use_filter = False
    # csv_parser("item.dat")
    inputs = {}
    for filename in file_list:
        print(f"Parsing file: {filename}")
        data, sk_range = csv_parser(filename, data_load_app[1])
        inputs[filename] = [data, sk_range]
        print("==================")
        # print(build_json(path, dataset[filename]))
    filtered_inputs = {}
    """
    filename = "catalog_sales.dat"
    filtered_inputs[filename] = filter_table(inputs[filename][0], np.arange(1000), 15)
    print("=====filtered=====")
    """
    # transfer to json
    logger = setup_logger(f"summary_{data_id}", f'./data/tpc-ds_{data_id}.log')
    for filename in file_list:
        # filter_table(inputs[filename][0], comb_range, 0)
        # filter out the rows are in certain range
        if use_filter:
            matrix_data = filter_table(inputs[filename][0], filter_range, sk_col)
        else:
            matrix_data = inputs[filename][0]
        print("=====filtered=====")
        data_dict = build_json(matrix_data, data_load_app[0][filename], size=10)
        for rid in range(len(data_dict)):
            logger.info(str(data_dict[rid]) + ",")
            if rid > 200:
                break
        print("---------")
    # comb_range = filter_range(inputs["date_dim.dat"][1][0], inputs["catalog_sales.dat"][1][0])  # inputs->skr_id[cid]
    # print(comb_range)


if __name__ == "__main__":
    # [80, 81, 85, 86, 89, 92, 93, 94, 98, 63, 64]
    # file_list = ["item.dat"]
    # data_id = 20
    # [23, 36, 47, 58, 72]
    for data_id in [93]:
        load_data(data_id)





