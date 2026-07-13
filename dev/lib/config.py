import math
from typing import Any, Literal

import lib
import lib.env

from . import datasets as devdatasets


# NOTE
# The following function is not used anywhere and is saved here only as a reference.
def _get_tabarena_batch_size(train_size: int) -> int:
    # The following series of conditions is taken from here:
    # https://github.com/autogluon/tabarena/blob/585cb490702c5c78eb5c0b700155aee20269936f/tabarena/benchmark/models/ag/tabm/tabm_model.py#L262C9-L272C20
    if train_size < 1_400:
        batch_size = 16
    elif train_size < 2_800:
        batch_size = 32
    elif train_size < 4_500:
        batch_size = 64
    elif train_size < 6_400:
        batch_size = 128
    elif train_size < 32_000:
        batch_size = 256
    elif train_size < 108_000:
        batch_size = 512
    else:
        batch_size = 1024

    # If the epoch size is less than 50 batches, reduce the batch size.
    if math.ceil(train_size / batch_size) < 50:
        batch_size = int(2 ** math.trunc(math.log2(train_size // 50)))
        # However, don't go below batch_size=16.
        return max(16, batch_size)
    else:
        return batch_size


del _get_tabarena_batch_size

_NN_BATCH_SIZES = {
    devdatasets.CHURN: 256,
    devdatasets.CALIFORNIA: 256,
    devdatasets.HOUSE: 256,
    devdatasets.ADULT: 256,
    devdatasets.DIAMOND: 512,
    devdatasets.OTTO: 512,
    devdatasets.HIGGS_SMALL: 512,
    devdatasets.BLACK_FRIDAY: 512,
    devdatasets.COVTYPE2: 1024,
    devdatasets.MICROSOFT: 1024,
    #
    devdatasets.TABRED_SBERBANK_HOUSING: 256,
    devdatasets.TABRED_ECOM_OFFERS: 1024,
    devdatasets.TABRED_MAPS_ROUTING: 1024,
    devdatasets.TABRED_HOMESITE_INSURANCE: 1024,
    devdatasets.TABRED_COOKING_TIME: 1024,
    devdatasets.TABRED_HOMECREDIT_DEFAULT: 1024,
    devdatasets.TABRED_DELIVERY_ETA: 1024,
    devdatasets.TABRED_WEATHER: 1024,
    #
    devdatasets.TABARENA_AIRFOIL_SELF_NOISE: 16,
    devdatasets.TABARENA_AMAZON_EMPLOYEE_ACCESS: 256,
    devdatasets.TABARENA_ANNEAL: 16,
    devdatasets.TABARENA_ANOTHER_DATASET_ON_USED_FIAT_500: 16,
    devdatasets.TABARENA_APSFAILURE: 512,
    devdatasets.TABARENA_BANK_MARKETING: 256,
    devdatasets.TABARENA_BANK_CUSTOMER_CHURN: 64,
    devdatasets.TABARENA_BIORESPONSE: 32,
    devdatasets.TABARENA_BLOOD_TRANSFUSION_SERVICE_CENTER: 16,
    devdatasets.TABARENA_CHURN: 32,
    devdatasets.TABARENA_COIL2000_INSURANCE_POLICIES: 64,
    devdatasets.TABARENA_CONCRETE_COMPRESSIVE_STRENGTH: 16,
    devdatasets.TABARENA_CREDIT_G: 16,
    devdatasets.TABARENA_CREDIT_CARD_CLIENTS_DEFAULT: 256,
    devdatasets.TABARENA_CUSTOMER_SATISFACTION_IN_AIRLINE: 512,
    devdatasets.TABARENA_DIABETES: 16,
    devdatasets.TABARENA_DIABETES130US: 512,
    devdatasets.TABARENA_DIAMONDS: 256,
    devdatasets.TABARENA_E_COMMERESHIPPINGDATA: 128,
    devdatasets.TABARENA_FITNESS_CLUB: 16,
    devdatasets.TABARENA_FOOD_DELIVERY_TIME: 256,
    devdatasets.TABARENA_GIVEMESOMECREDIT: 512,
    devdatasets.TABARENA_HAZELNUT_SPREAD_CONTAMINANT_DETECTION: 16,
    devdatasets.TABARENA_HEALTHCARE_INSURANCE_EXPENSES: 16,
    devdatasets.TABARENA_HELOC: 64,
    devdatasets.TABARENA_HIVA_AGNOSTIC: 32,
    devdatasets.TABARENA_HOUSES: 128,
    devdatasets.TABARENA_HR_ANALYTICS_JOB_CHANGE_OF_DATA_SCIENTISTS: 128,
    devdatasets.TABARENA_IN_VEHICLE_COUPON_RECOMMENDATION: 128,
    devdatasets.TABARENA_IS_THIS_A_GOOD_CUSTOMER: 16,
    devdatasets.TABARENA_KDDCUP09_APPETENCY: 256,
    devdatasets.TABARENA_MARKETING_CAMPAIGN: 16,
    devdatasets.TABARENA_MATERNAL_HEALTH_RISK: 16,
    devdatasets.TABARENA_MIAMI_HOUSING: 128,
    devdatasets.TABARENA_NATICUSDROID: 64,
    devdatasets.TABARENA_ONLINE_SHOPPERS_INTENTION: 128,
    devdatasets.TABARENA_PHYSIOCHEMICAL_PROTEIN: 256,
    devdatasets.TABARENA_POLISH_COMPANIES_BANKRUPTCY: 64,
    devdatasets.TABARENA_QSAR_BIODEG: 16,
    devdatasets.TABARENA_QSAR_TID_11: 64,
    devdatasets.TABARENA_QSAR_FISH_TOXICITY: 16,
    devdatasets.TABARENA_SDSS17: 512,
    devdatasets.TABARENA_SEISMIC_BUMPS: 16,
    devdatasets.TABARENA_SPLICE: 32,
    devdatasets.TABARENA_STUDENTS_DROPOUT_AND_ACADEMIC_SUCCESS: 32,
    devdatasets.TABARENA_SUPERCONDUCTIVITY: 128,
    devdatasets.TABARENA_TAIWANESE_BANKRUPTCY_PREDICTION: 64,
    devdatasets.TABARENA_WEBSITE_PHISHING: 16,
    devdatasets.TABARENA_WINE_QUALITY: 64,
    devdatasets.TABARENA_MIC: 16,
    devdatasets.TABARENA_JM1: 128,
}


def get_nn_batch_size(dataset: str) -> int:
    return _NN_BATCH_SIZES[dataset]


_NN_NUM_POLICY = {
    x: 'noisy-quantile'
    for x in _NN_BATCH_SIZES.keys()
    if x
    not in {
        # The 'noisy-quantile' normalization works poorly for the OTTO dataset.
        devdatasets.OTTO,
        # The following TabReD datasets are already normalized.
        devdatasets.TABRED_MAPS_ROUTING,
        devdatasets.TABRED_COOKING_TIME,
        devdatasets.TABRED_DELIVERY_ETA,
    }
}


def make_data_config(
    dataset: str,
    *,
    num_policy: None | Literal['nn'] = 'nn',
    cat_policy: str = 'ordinal',
    cache: bool = False,
) -> dict[str, Any]:
    data_config = {}

    dataset_path = lib.env.get_data_dir() / dataset
    if not dataset_path.exists():
        raise RuntimeError(f'The dataset does not exist: {dataset_path}')
    data_config['path'] = str(dataset_path.relative_to(lib.env.get_project_dir()))

    if num_policy == 'nn':
        num_policy_ = _NN_NUM_POLICY.get(dataset)
        if num_policy_ is not None:
            data_config['num_policy'] = num_policy_
    data_config['extract_bin_from_num'] = True
    data_config['bin_policy'] = 'convert-to-cat'
    if dataset_path.joinpath('x_cat.npy').exists():
        data_config['cat_policy'] = cat_policy

    data_config['cache'] = cache

    return data_config
