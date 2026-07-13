from pathlib import Path
from typing import Any

CHURN = 'churn'
CALIFORNIA = 'california'
HOUSE = 'house'
ADULT = 'adult'
DIAMOND = 'diamond'
OTTO = 'otto'
HIGGS_SMALL = 'higgs-small'
BLACK_FRIDAY = 'black-friday'
COVTYPE2 = 'covtype2'
MICROSOFT = 'microsoft'
INDEPENDENT_DATASETS = [
    CHURN,
    CALIFORNIA,
    HOUSE,
    ADULT,
    DIAMOND,
    OTTO,
    HIGGS_SMALL,
    BLACK_FRIDAY,
    COVTYPE2,
    MICROSOFT,
]

TABRED_SBERBANK_HOUSING = 'tabred/sberbank-housing'
TABRED_ECOM_OFFERS = 'tabred/ecom-offers'
TABRED_MAPS_ROUTING = 'tabred/maps-routing'
TABRED_HOMESITE_INSURANCE = 'tabred/homesite-insurance'
TABRED_COOKING_TIME = 'tabred/cooking-time'
TABRED_HOMECREDIT_DEFAULT = 'tabred/homecredit-default'
TABRED_DELIVERY_ETA = 'tabred/delivery-eta'
TABRED_WEATHER = 'tabred/weather'
TABRED_DATASETS = [
    TABRED_SBERBANK_HOUSING,
    TABRED_ECOM_OFFERS,
    TABRED_MAPS_ROUTING,
    TABRED_HOMESITE_INSURANCE,
    TABRED_COOKING_TIME,
    TABRED_HOMECREDIT_DEFAULT,
    TABRED_DELIVERY_ETA,
    TABRED_WEATHER,
]

# fmt: off
TABARENA_AIRFOIL_SELF_NOISE = 'tabarena/airfoil_self_noise'
TABARENA_AMAZON_EMPLOYEE_ACCESS = 'tabarena/Amazon_employee_access'
TABARENA_ANNEAL = 'tabarena/anneal'
TABARENA_ANOTHER_DATASET_ON_USED_FIAT_500 = 'tabarena/Another-Dataset-on-used-Fiat-500'
TABARENA_APSFAILURE = 'tabarena/APSFailure'
TABARENA_BANK_MARKETING = 'tabarena/bank-marketing'
TABARENA_BANK_CUSTOMER_CHURN = 'tabarena/Bank_Customer_Churn'
TABARENA_BIORESPONSE = 'tabarena/Bioresponse'
TABARENA_BLOOD_TRANSFUSION_SERVICE_CENTER = 'tabarena/blood-transfusion-service-center'
TABARENA_CHURN = 'tabarena/churn'
TABARENA_COIL2000_INSURANCE_POLICIES = 'tabarena/coil2000_insurance_policies'
TABARENA_CONCRETE_COMPRESSIVE_STRENGTH = 'tabarena/concrete_compressive_strength'
TABARENA_CREDIT_G = 'tabarena/credit-g'
TABARENA_CREDIT_CARD_CLIENTS_DEFAULT = 'tabarena/credit_card_clients_default'
TABARENA_CUSTOMER_SATISFACTION_IN_AIRLINE = 'tabarena/customer_satisfaction_in_airline'
TABARENA_DIABETES = 'tabarena/diabetes'
TABARENA_DIABETES130US = 'tabarena/Diabetes130US'
TABARENA_DIAMONDS = 'tabarena/diamonds'
TABARENA_E_COMMERESHIPPINGDATA = 'tabarena/E-CommereShippingData'
TABARENA_FITNESS_CLUB = 'tabarena/Fitness_Club'
TABARENA_FOOD_DELIVERY_TIME = 'tabarena/Food_Delivery_Time'
TABARENA_GIVEMESOMECREDIT = 'tabarena/GiveMeSomeCredit'
TABARENA_HAZELNUT_SPREAD_CONTAMINANT_DETECTION = 'tabarena/hazelnut-spread-contaminant-detection'  # noqa: E501
TABARENA_HEALTHCARE_INSURANCE_EXPENSES = 'tabarena/healthcare_insurance_expenses'
TABARENA_HELOC = 'tabarena/heloc'
TABARENA_HIVA_AGNOSTIC = 'tabarena/hiva_agnostic'
TABARENA_HOUSES = 'tabarena/houses'
TABARENA_HR_ANALYTICS_JOB_CHANGE_OF_DATA_SCIENTISTS = 'tabarena/HR_Analytics_Job_Change_of_Data_Scientists'  # noqa: E501
TABARENA_IN_VEHICLE_COUPON_RECOMMENDATION = 'tabarena/in_vehicle_coupon_recommendation'
TABARENA_IS_THIS_A_GOOD_CUSTOMER = 'tabarena/Is-this-a-good-customer'
TABARENA_KDDCUP09_APPETENCY = 'tabarena/kddcup09_appetency'
TABARENA_MARKETING_CAMPAIGN = 'tabarena/Marketing_Campaign'
TABARENA_MATERNAL_HEALTH_RISK = 'tabarena/maternal_health_risk'
TABARENA_MIAMI_HOUSING = 'tabarena/miami_housing'
TABARENA_NATICUSDROID = 'tabarena/NATICUSdroid'
TABARENA_ONLINE_SHOPPERS_INTENTION = 'tabarena/online_shoppers_intention'
TABARENA_PHYSIOCHEMICAL_PROTEIN = 'tabarena/physiochemical_protein'
TABARENA_POLISH_COMPANIES_BANKRUPTCY = 'tabarena/polish_companies_bankruptcy'
TABARENA_QSAR_BIODEG = 'tabarena/qsar-biodeg'
TABARENA_QSAR_TID_11 = 'tabarena/QSAR-TID-11'
TABARENA_QSAR_FISH_TOXICITY = 'tabarena/QSAR_fish_toxicity'
TABARENA_SDSS17 = 'tabarena/SDSS17'
TABARENA_SEISMIC_BUMPS = 'tabarena/seismic-bumps'
TABARENA_SPLICE = 'tabarena/splice'
TABARENA_STUDENTS_DROPOUT_AND_ACADEMIC_SUCCESS = 'tabarena/students_dropout_and_academic_success'  # noqa: E501
TABARENA_SUPERCONDUCTIVITY = 'tabarena/superconductivity'
TABARENA_TAIWANESE_BANKRUPTCY_PREDICTION = 'tabarena/taiwanese_bankruptcy_prediction'
TABARENA_WEBSITE_PHISHING = 'tabarena/website_phishing'
TABARENA_WINE_QUALITY = 'tabarena/wine_quality'
TABARENA_MIC = 'tabarena/MIC'
TABARENA_JM1 = 'tabarena/jm1'
# fmt: on
TABARENA_DATASETS = [
    TABARENA_AIRFOIL_SELF_NOISE,
    TABARENA_AMAZON_EMPLOYEE_ACCESS,
    TABARENA_ANNEAL,
    TABARENA_ANOTHER_DATASET_ON_USED_FIAT_500,
    TABARENA_APSFAILURE,
    TABARENA_BANK_MARKETING,
    TABARENA_BANK_CUSTOMER_CHURN,
    TABARENA_BIORESPONSE,
    TABARENA_BLOOD_TRANSFUSION_SERVICE_CENTER,
    TABARENA_CHURN,
    TABARENA_COIL2000_INSURANCE_POLICIES,
    TABARENA_CONCRETE_COMPRESSIVE_STRENGTH,
    TABARENA_CREDIT_G,
    TABARENA_CREDIT_CARD_CLIENTS_DEFAULT,
    TABARENA_CUSTOMER_SATISFACTION_IN_AIRLINE,
    TABARENA_DIABETES,
    TABARENA_DIABETES130US,
    TABARENA_DIAMONDS,
    TABARENA_E_COMMERESHIPPINGDATA,
    TABARENA_FITNESS_CLUB,
    TABARENA_FOOD_DELIVERY_TIME,
    TABARENA_GIVEMESOMECREDIT,
    TABARENA_HAZELNUT_SPREAD_CONTAMINANT_DETECTION,
    TABARENA_HEALTHCARE_INSURANCE_EXPENSES,
    TABARENA_HELOC,
    TABARENA_HIVA_AGNOSTIC,
    TABARENA_HOUSES,
    TABARENA_HR_ANALYTICS_JOB_CHANGE_OF_DATA_SCIENTISTS,
    TABARENA_IN_VEHICLE_COUPON_RECOMMENDATION,
    TABARENA_IS_THIS_A_GOOD_CUSTOMER,
    TABARENA_KDDCUP09_APPETENCY,
    TABARENA_MARKETING_CAMPAIGN,
    TABARENA_MATERNAL_HEALTH_RISK,
    TABARENA_MIAMI_HOUSING,
    TABARENA_NATICUSDROID,
    TABARENA_ONLINE_SHOPPERS_INTENTION,
    TABARENA_PHYSIOCHEMICAL_PROTEIN,
    TABARENA_POLISH_COMPANIES_BANKRUPTCY,
    TABARENA_QSAR_BIODEG,
    TABARENA_QSAR_TID_11,
    TABARENA_QSAR_FISH_TOXICITY,
    TABARENA_SDSS17,
    TABARENA_SEISMIC_BUMPS,
    TABARENA_SPLICE,
    TABARENA_STUDENTS_DROPOUT_AND_ACADEMIC_SUCCESS,
    TABARENA_SUPERCONDUCTIVITY,
    TABARENA_TAIWANESE_BANKRUPTCY_PREDICTION,
    TABARENA_WEBSITE_PHISHING,
    TABARENA_WINE_QUALITY,
    TABARENA_MIC,
    TABARENA_JM1,
]

assert not (set(INDEPENDENT_DATASETS) & set(TABRED_DATASETS))
assert not (set(INDEPENDENT_DATASETS) & set(TABARENA_DATASETS))
assert not (set(TABRED_DATASETS) & set(TABARENA_DATASETS))


def load_extended_info(dataset_dir: str | Path, split_id) -> dict[str, Any]:
    import lib.data

    dataset_dir = lib.data._check_dataset_dir(dataset_dir)

    info = lib.data.load_info(dataset_dir)
    data = lib.data.load_data(dataset_dir, split_id)

    some_subdata = next(iter(data.values()))
    for part, value in some_subdata.items():
        info[f'{part}_size'] = len(value)

    info['n_features'] = 0
    for key, subdata in data.items():
        if key.startswith('x_'):
            some_value = next(iter(subdata.values()))
            n_features = some_value.shape[1]
            info[f'n_{key.removeprefix("x_")}_features'] = n_features
            info['n_features'] += n_features

    return info
