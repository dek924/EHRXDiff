import os
import glob
import json
import logging

import numpy as np
import pandas as pd

import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from ehrs import EHR, register_ehr

logger = logging.getLogger(__name__)

@register_ehr("mimicivfiltered")
class MIMICIVFiltered(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ehr_name = "mimiciv"

        if self.data_dir is None:
            self.data_dir = os.path.join(self.cache_dir, self.ehr_name)

            if not os.path.exists(self.data_dir):
                logger.info(
                    "Data is not found so try to download from the internet. "
                    "Note that this is a restricted-access resource. "
                    "Please log in to physionet.org with a credentialed user."
                )
                self.download_ehr_from_url(
                    url="https://physionet.org/files/mimiciv/2.0/",
                    dest=self.data_dir
                )

        logger.info("Data directory is set to {}".format(self.data_dir))

        if self.ccs_path is None:
            self.ccs_path = os.path.join(self.cache_dir, "ccs_multi_dx_tool_2015.csv")

            if not os.path.exists(self.ccs_path):
                logger.info(
                    "`ccs_multi_dx_tool_2015.csv` is not found so try to download from the internet."
                )
                self.download_ccs_from_url(self.cache_dir)

        if self.gem_path is None:
            self.gem_path = os.path.join(self.cache_dir, "icd10cmtoicd9gem.csv")

            if not os.path.exists(self.gem_path):
                logger.info(
                    "`icd10cmtoicd9gem.csv` is not found so try to download from the internet."
                )
                self.download_icdgem_from_url(self.cache_dir)

        if self.ext is None:
            self.ext = self.infer_data_extension()

        self._icustay_fname = "icu/icustays" + self.ext
        self._patient_fname = "hosp/patients" + self.ext
        self._admission_fname = "hosp/admissions" + self.ext
        self._diagnosis_fname = "hosp/diagnoses_icd" + self.ext
        self._prescription_fname = "hosp/prescriptions" + self.ext
        self._transfer_fname = "hosp/transfers" + self.ext

        self.tables = [
            {
                "fname": "hosp/labevents" + self.ext,
                "timestamp": "charttime",
                "timeoffsetunit": "abs",
                "exclude": ["labevent_id", "storetime", "subject_id", "order_provider_id", "specimen_id"],
                "code": ["itemid"],
                "desc": ["hosp/d_labitems" + self.ext],
                "desc_key": ["label"],
                "filter_key": "itemid",
                "filter_value": self.get_filtered_items("labevents", "hosp/d_labitems" + self.ext),
            },
            {
                "fname": "hosp/prescriptions" + self.ext,
                "timestamp": "starttime",
                "timeoffsetunit": "abs",
                "exclude": [
                    "gsn",
                    "ndc",
                    "subject_id",
                    "order_provider_id",
                    "pharmacy_id",
                    "poe_id",
                    "poe_seq",
                    "formulary_drug_cd",
                    "stoptime",
                ],
                "filter_key": "drug",
                "filter_value": self.get_filtered_items("prescriptions", None),
            },
            {
                "fname": "icu/inputevents" + self.ext,
                "timestamp": "starttime",
                "timeoffsetunit": "abs",
                "exclude": [
                    "endtime",
                    "storetime",
                    "orderid",
                    "linkorderid",
                    "subject_id",
                    "caregiver_id",
                    "continueinnextdept",
                    "statusdescription",
                ],
                "code": ["itemid"],
                "desc": ["icu/d_items" + self.ext],
                "desc_key": ["label"],
            },
        ]

        if cfg.use_more_tables:
            self.tables+=[
                {
                    "fname": "icu/chartevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "storetime",
                        "subject_id",
                        "caregiver_id",
                    ],
                    "code": ["itemid"],
                    "desc": ["icu/d_items" + self.ext],
                    "desc_key": ["label"],
                    "filter_key": "itemid",
                    "filter_value": self.get_filtered_items("chartevents", "icu/d_items" + self.ext),
                },
                {
                    "fname": "icu/outputevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "storetime",
                        "subject_id",
                        "caregiver_id",
                    ],
                    "code": ["itemid"],
                    "desc": ["icu/d_items" + self.ext],
                    "desc_key": ["label"],
                },
                {
                    "fname": "hosp/microbiologyevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": ["chartdate", "storetime", "storedate", "subject_id", "order_provider_id", "microevent_id", "micro_specimen_id", "spec_itemid", "test_itemid", "org_itemid", "ab_itemid"],
                    "filter_key": "spec_type_desc",
                    "filter_value": [
                        "SPUTUM",
                        "BRONCHOALVEOLAR LAVAGE",
                        "PLEURAL FLUID",
                        "Rapid Respiratory Viral Screen & Culture",
                        "BRONCHIAL WASHINGS",
                        "THROAT FOR STREP",
                        "Mini-BAL",
                        "ASPIRATE",
                        "THROAT CULTURE",
                        "BRONCHIAL BRUSH",
                        "CHORIONIC VILLUS SAMPLE",
                        "TRACHEAL ASPIRATE",
                        "RAPID RESPIRATORY VIRAL ANTIGEN TEST",
                        "BRONCHIAL BRUSH - PROTECTED",
                        "THROAT"
                        ] 
                },
                {
                    "fname": "icu/procedureevents" + self.ext,
                    "timestamp": "starttime",
                    "timeoffsetunit": "abs",
                    "exclude": ["storetime", "endtime", "subject_id", "caregiver_id", "orderid", "linkorderid", "continueinnextdept", "statusdescription"],
                    "code": ["itemid"],
                    "desc": ["icu/d_items" + self.ext],
                    "desc_key": ["label"],
                    "filter_key": "ordercategoryname",
                    "filter_value": [
                        "Procedures",
                        "Imaging",
                        "Invasive Lines",
                        "Ventilation",
                        "Intubation/Extubation",
                        "Significant Events",
                        "17 - Inhaled Meds",
                        ] 
                },
            ]

        self.disch_map_dict = {
            "ACUTE HOSPITAL": "Other",
            "AGAINST ADVICE": "Other",
            "ASSISTED LIVING": "Other",
            "CHRONIC/LONG TERM ACUTE CARE": "Other",
            "HEALTHCARE FACILITY": "Other",
            "HOME": "Home",
            "HOME HEALTH CARE": "Home",
            "HOSPICE": "Other",
            "IN_ICU_MORTALITY": "IN_ICU_MORTALITY",
            "OTHER FACILITY": "Other",
            "PSYCH FACILITY": "Other",
            "REHAB": "Rehabilitation",
            "SKILLED NURSING FACILITY": "Skilled Nursing Facility",
            "Death": "Death",
        }

        self._icustay_key = "stay_id"
        self._hadm_key = "hadm_id"
        self._patient_key = "subject_id"
        self._image_key = "dicom_id"
        if self.use_image:
            self._icustay_key = "dicom_id"

        self._determine_first_icu = "INTIME"

    def get_filtered_items(self, table_name, dict_fname):
        if table_name == "prescriptions":
            # load json file
            with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"selected_items/selected_prescription_dict.json"), "r") as f:
                d_items = json.load(f)
            
            prescriptions_list = d_items["antibiotic"] + d_items["cardiac_lung"] 
            prescriptions_list = [_item for _item in prescriptions_list if ("tobramycin" not in _item.lower()) and ("coq10" not in _item.lower())]
            print(f"# of filtered prescriptions : {len(prescriptions_list)}")
            return prescriptions_list
        elif table_name == "inputevents":
            d_items = pd.read_csv(os.path.join(self.data_dir, "icu/d_items.csv.gz"))
            inputevents = pd.read_csv(os.path.join(self.data_dir, "icu/inputevents.csv.gz"), usecols=["itemid", "ordercategoryname"])
            inputevents_list = inputevents[inputevents.ordercategoryname.isin(["08-Antibiotics (IV)", "09-Antibiotics (Non IV)"])].itemid.unique().tolist()

            input_medicaation_list = ['Furosemide (Lasix)',
                                    'Heparin Sodium',
                                    'Heparin Sodium (Prophylaxis)',
                                    'Hydralazine',
                                    'Enoxaparin (Lovenox)',
                                    'Dopamine',
                                    'Nitroglycerin',
                                    'Phenylephrine',
                                    'Potassium Chloride',
                                    'KCL (Bolus)',
                                    'Coumadin (Warfarin)',
                                    'Norepinephrine',
                                    'Metoprolol',
                                    'Epinephrine',
                                    'Amiodarone',
                                    'Amiodarone 600/500',
                                    'Diltiazem',
                                    'Vasopressin',
                                    'Digoxin (Lanoxin)',
                                    'Esmolol',
                                    'Sodium Bicarbonate 8.4%',
                                    'Sodium Bicarbonate 8.4% (Amp)',
                                    'Nutren Pulmonary (Full)',
                                    'Dexmedetomidine (Precedex)',
                                    'Nicardipine',
                                    'Nicardipine 40mg/200',
                                    'Labetalol',
                                    'Alteplase (TPA)',
                                    'Protamine sulfate',
                                    'Heparin Sodium (Impella)',
                                    'Amiodarone',
                                    'Amiodarone 450/250',
                                    'Phenylephrine (50/250)',
                                    'Phenylephrine (200/250)',
                                    'Milrinone',
                                    'Dobutamine',
                                    'Pulmocare (Full)',
                                    'Mannitol',
                                    'Argatroban',
                                    'Nitroprusside',
                                    'Procainamide',
                                    'Verapamil',
                                    'Fondaparinux',
                                    'Clevidipine (Cleviprex)',
                                    'Tirofiban (Aggrastat)',
                                    'Epinephrine',
                                    'Angiotensin II (Giapreza)',
                                    'Isuprel',
                                    'Factor VIIa',
                                    'Abciximab (Reopro)',
                                    'Nesiritide',
                                    'Cell Saver',
                                    'Hetastarch (Hespan) 6%',
                                    'Treprostinil (Remodulin)',
                                    'Epoprostenol (Veletri)',
                                    'Aminophylline',
                                    'Bivalirudin (Angiomax) (Impella)',
                                    'Pulmocare (1/2)',
                                    'Pulmocare (1/4)',
                                    'Lepirudin',
                                    'Lipids 10%']

            inputevents_list += d_items[d_items.label.isin(input_medicaation_list)].itemid.unique().tolist()
            print(f"# of filtered inputevents : {len(inputevents_list)}")
            return inputevents_list
        else:
            assert dict_fname in ["icu/d_items" + self.ext, "hosp/d_labitems" + self.ext]
            d_items = pd.read_csv(os.path.join(self.data_dir, dict_fname))
            if table_name == "chartevents":
                d_items = d_items[d_items.linksto == table_name]
                useful_category = [
                    "Access Lines - Invasive",
                    "Respiratory",
                    "Labs",
                    "Pain/Sedation",
                    "MD Progress Note",
                    "Routine Vital Signs",
                    "Pulmonary",
                    "ECMO",
                    "Alarms",
                    "Hemodynamics",
                    "Cardiovascular",
                    "Impella",
                    "Cardiovascular (Pacer Data)",
                    "General",
                    "Centrimag",
                    "Durable VAD",
                    "Cardiovascular (Pulses)",
                    "Tandem Heart",
                    "NICOM",
                    "IABP",
                    "PiCCO",
                    "Heartware",
                    "PA Line Insertion",
                    "Case Management"
                ]
                d_items = d_items[d_items.category.isin(useful_category)]
                print(f"# of filtered chartevents : {len(d_items)}")
                return d_items["itemid"].tolist()
            elif table_name == "labevents":
                d_items = d_items[(d_items.category == "Blood Gas") | (d_items.label.isin(["C-Reactive Protein", "High-Sensitivity CRP"]))]
                print(f"# of filtered labevents : {len(d_items)}")
                return d_items["itemid"].tolist()
            else:
                raise NotImplementedError
            
    def build_cohorts(self, cached=False):
        if self.use_admission_table:
            icustays = pd.read_csv(os.path.join(self.data_dir, self.admission_fname))
            icustays = self.make_compatible_admission(icustays)
        else:
            icustays = pd.read_csv(os.path.join(self.data_dir, self.icustay_fname))
            icustays = self.make_compatible(icustays)
        self.icustays = icustays

        cohorts = super().build_cohorts(icustays, cached=cached)
        print(cohorts.shape)
        return cohorts

    def prepare_tasks(self, cohorts, spark, cached=False):
        if cached:
            labeled_cohorts = self.load_from_cache(self.ehr_name + ".cohorts.labeled")
            if labeled_cohorts is not None:
                return labeled_cohorts

        labeled_cohorts = super().prepare_tasks(cohorts, spark, cached)

        if self.diagnosis:
            logger.info(
                "Start labeling cohorts for diagnosis prediction."
            )

            # define diagnosis prediction task
            diagnoses = pd.read_csv(os.path.join(self.data_dir, self.diagnosis_fname))

            diagnoses = self.icd10toicd9(diagnoses)

            ccs_dx = pd.read_csv(self.ccs_path)
            ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.strip()
            ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1]
            lvl1 = {
                x: int(y)-1 for _, (x, y) in ccs_dx[["'ICD-9-CM CODE'", "'CCS LVL 1'"]].iterrows()
            }

            diagnoses['diagnosis'] = diagnoses['icd_code_converted'].map(lvl1)

            diagnoses = diagnoses[(diagnoses['diagnosis'].notnull()) & (diagnoses['diagnosis']!=14)]
            diagnoses.loc[diagnoses['diagnosis']>=14, 'diagnosis'] -= 1
            diagnoses = diagnoses.groupby(self.hadm_key)['diagnosis'].agg(lambda x: list(set(x))).to_frame()

            labeled_cohorts = labeled_cohorts.merge(diagnoses, on=self.hadm_key, how='inner')

            logger.info("Done preparing diagnosis prediction for the given cohorts")

            self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")

        if self.prescription:
            logger.info(
                "Start labeling cohorts for prescription prediction."
            )

            # define prescription prediction task
            prescriptions = pd.read_csv(os.path.join(self.data_dir, self._prescription_fname))
            prescriptions = prescriptions.drop(columns=["gsn", "ndc", "subject_id", "pharmacy_id", "poe_id", "poe_seq", "formulary_drug_cd", "stoptime"])
            prescriptions["starttime"] = pd.to_datetime(prescriptions["starttime"], utc=True)
            prescriptions = prescriptions[(prescriptions['drug'].notnull())]

            _merge = prescriptions.merge(labeled_cohorts[[self.hadm_key, "INTIME", "OUTTIME"]], on=self.hadm_key, how="inner").copy()
            _merge["_OUTTIME"] = (_merge["starttime"] - _merge["INTIME"]).dt.total_seconds() // 60
            _merge = _merge[(_merge._OUTTIME > 0) & (_merge._OUTTIME <= _merge.OUTTIME)]
            if self.rolling_from_last:
                _merge = _merge[(self.obs_size * 60 <= _merge["_OUTTIME"]) & (self.obs_size * 60 >= _merge["_OUTTIME"])]
            _merge = _merge.groupby(self.hadm_key)['drug'].agg(lambda x: list(set(x))).to_frame()

            labeled_cohorts = labeled_cohorts.merge(_merge, on=self.hadm_key, how='inner')
            
            logger.info("Done preparing prescription prediction for the given cohorts")
            
            labeled_cohorts.to_csv(os.path.join(self.dest, f'{self.ehr_name}_cohort_labeled.csv'), index=False)
            self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")

        if self.bilirubin or self.platelets or self.creatinine or self.wbc or self.hb or self.bicarbonate or self.sodium or self.antibiotics:
            logger.info(
                "Start labeling cohorts for clinical task prediction."
            )

            labeled_cohorts = spark.createDataFrame(labeled_cohorts)
            
            if self.bilirubin:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "bilirubin", spark)

            if self.platelets:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "platelets", spark)

            if self.creatinine:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "creatinine", spark)
            
            if self.wbc:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "wbc", spark)
            
            if self.hb:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "hb", spark)
            
            if self.bicarbonate:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "bicarbonate", spark)
            
            if self.sodium:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "sodium", spark)
            
            if self.antibiotics:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "antibiotics", spark)

            logger.info("Done preparing clinical task prediction for the given cohorts")
        
        if not isinstance(labeled_cohorts, pd.DataFrame):
            labeled_cohorts = labeled_cohorts.toPandas()

        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")
        return labeled_cohorts

    def make_compatible(self, icustays):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patient_fname))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admission_fname))
        if self.use_image:
            img_meta = pd.read_csv(os.path.join(self.img_data_dir, "mimic-cxr-2.0.0-metadata.csv"))
            img_meta = img_meta[['dicom_id', 'subject_id', 'study_id', 'StudyDate', 'StudyTime']]
            img_meta['StudyTime'] = img_meta['StudyTime'].apply(lambda x: f'{int(float(x)):06}')
            img_meta['StudyDateTime'] = pd.to_datetime(img_meta['StudyDate'].astype(str) + ' ' + img_meta['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")
            img_meta = img_meta[['dicom_id', 'subject_id', 'study_id', 'StudyDateTime']]

        # prepare icustays according to the appropriate format
        icustays = icustays.rename(columns={
            "los": "LOS",
            "intime": "INTIME",
            "outtime": "OUTTIME",
        })
        admissions = admissions.rename(columns={
            "dischtime": "DISCHTIME",
        })

        icustays = icustays[icustays["first_careunit"] == icustays["last_careunit"]]
        icustays.loc[:, "INTIME"] = pd.to_datetime(
            icustays["INTIME"], utc=True
        )
        icustays.loc[:, "OUTTIME"] = pd.to_datetime(
            icustays["OUTTIME"], utc=True
        )

        icustays = icustays.merge(patients, on="subject_id", how="left")
        icustays["AGE"] = (
            icustays["INTIME"].dt.year
            - icustays["anchor_year"]
            + icustays["anchor_age"]
        )

        icustays = icustays.merge(
            admissions[
                [self.hadm_key, "discharge_location", "deathtime", "DISCHTIME"]
            ],
            how="left",
            on=self.hadm_key,
        )

        icustays["discharge_location"].replace("DIED", "Death", inplace=True)
        icustays["DISCHTIME"] = pd.to_datetime(
            icustays["DISCHTIME"], utc=True
        )

        icustays["IN_ICU_MORTALITY"] = (
            (icustays["INTIME"] < icustays["DISCHTIME"])
            & (icustays["DISCHTIME"] <= icustays["OUTTIME"])
            & (icustays["discharge_location"] == "Death")
        )
        icustays["discharge_location"] = icustays["discharge_location"].map(self.disch_map_dict)
        icustays.rename(columns={"discharge_location": "HOS_DISCHARGE_LOCATION"}, inplace=True)

        icustays["DISCHTIME"] = (icustays["DISCHTIME"] - icustays["INTIME"]).dt.total_seconds() // 60
        icustays["OUTTIME"] = (icustays["OUTTIME"] - icustays["INTIME"]).dt.total_seconds() // 60
        
        if self.use_image:
            icustays = icustays.merge(img_meta[[self.patient_key, self.image_key, "StudyDateTime"]], on=self.patient_key, how="inner")
            icustays = icustays[~icustays[self.image_key].isna()]
            icustays["IMG_OUTTIME"] = (icustays["StudyDateTime"] - icustays["INTIME"]).dt.total_seconds() // 60
            icustays = icustays[(icustays.IMG_OUTTIME > 0) & (icustays.IMG_OUTTIME <= icustays.OUTTIME)]
            icustays = icustays.drop(columns=['OUTTIME', 'StudyDateTime'])
            icustays = icustays.rename(columns={"IMG_OUTTIME": "OUTTIME"})

        return icustays

    def make_compatible_admission(self, admissions):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patient_fname))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admission_fname))
        admissions = admissions[[self.patient_key, self.hadm_key, "discharge_location", "deathtime", "admittime", "dischtime"]]
        print(f"Load admission table: ", admissions.shape)

        admissions = admissions.rename(
            columns={
                "admittime": "ADMITTIME",
                "dischtime": "DISCHTIME",
            }
        )
        admissions["LOS"] = -1
        admissions["IN_ICU_MORTALITY"] = -1
        admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], utc=True)
        admissions["DISCHTIME"] = pd.to_datetime(admissions["DISCHTIME"], utc=True)

        admissions = admissions.merge(patients, on="subject_id", how="left")
        admissions["AGE"] = admissions["ADMITTIME"].dt.year - admissions["anchor_year"] + admissions["anchor_age"]
        admissions = admissions[admissions["AGE"] >= self.min_age]
        print(f"Use only adult patients (min age: {self.min_age}): ", admissions.shape)

        admissions["discharge_location"].replace("DIED", "Death", inplace=True)
        admissions["discharge_location"] = admissions["discharge_location"].map(self.disch_map_dict)
        admissions.rename(columns={"discharge_location": "HOS_DISCHARGE_LOCATION"}, inplace=True)

        admissions["DISCHTIME"] = (admissions["DISCHTIME"] - admissions["ADMITTIME"]).dt.total_seconds() // 60
        print(admissions.shape)

        if self.use_image:
            cxr_meta = pd.read_csv(os.path.join(self.img_data_dir, "mimic-cxr-2.0.0-metadata.csv"))
            cxr_meta = cxr_meta[["dicom_id", "subject_id", "study_id", "StudyDate", "StudyTime", "ViewPosition"]]
            print("Load mimic cxr metadata: ", cxr_meta.shape)

            # Use only frontal view image
            cxr_meta = cxr_meta[cxr_meta.ViewPosition.isin(["PA", "AP"])]
            cxr_meta = cxr_meta.drop(columns=["ViewPosition"])
            print("Use only frontal view image: ", cxr_meta.shape)

            # Convert StudyDate and StudyTime to datetime
            cxr_meta["StudyTime"] = cxr_meta["StudyTime"].apply(lambda x: f"{int(float(x)):06}")
            cxr_meta["StudyDateTime"] = pd.to_datetime(cxr_meta["StudyDate"].astype(str) + " " + cxr_meta["StudyTime"].astype(str), format="%Y%m%d %H%M%S", utc=True)
            cxr_meta = cxr_meta.drop(columns=["StudyDate", "StudyTime"])

            # Get StudyOrder
            _cxr_meta = cxr_meta.copy()
            _cxr_meta = _cxr_meta.sort_values(by=["subject_id", "study_id", "StudyDateTime"])
            _cxr_meta = _cxr_meta.drop_duplicates(subset=["subject_id", "study_id"], keep="first").copy()
            _cxr_meta["StudyOrder"] = _cxr_meta.groupby(["subject_id"])["StudyDateTime"].rank(method="dense")
            cxr_meta["StudyOrder"] = cxr_meta["study_id"].map(_cxr_meta[["study_id", "StudyOrder"]].set_index("study_id")["StudyOrder"])
            cxr_meta = cxr_meta[["dicom_id", "subject_id", "study_id", "StudyDateTime", "StudyOrder"]]

            # Merge cxr_meta
            patient_cxr_info = admissions.merge(cxr_meta, on=self.patient_key, how="left").copy()
            print("Merge cxr metadata: ", patient_cxr_info.shape)

            # Use patient with at least one image
            patient_cxr_info = patient_cxr_info[~patient_cxr_info[self.image_key].isna()]
            print("Use patient with at least one image: ", patient_cxr_info.shape)

            # Use only images taken within hospital stay
            patient_cxr_info["OUTTIME"] = (patient_cxr_info["StudyDateTime"] - patient_cxr_info["ADMITTIME"]).dt.total_seconds() / 60
            patient_cxr_info = patient_cxr_info[(patient_cxr_info.OUTTIME > 0) & (patient_cxr_info.OUTTIME <= patient_cxr_info.DISCHTIME)]

            patient_cxr_info = patient_cxr_info.sort_values(by=["subject_id", "StudyDateTime"])

            # Get previous images
            def _get_img_history(df):
                df = df.sort_values("StudyDateTime")
                df["prev_dicom_id"] = [df["dicom_id"].values[:i].tolist() for i in range(len(df))]
                return df

            patient_cxr_info = patient_cxr_info.groupby(["subject_id", "hadm_id"]).apply(_get_img_history).reset_index(drop=True)

            # Use only patient with previous image
            patient_cxr_info = patient_cxr_info[patient_cxr_info.prev_dicom_id.str.len() > 0]
            patient_cxr_info = patient_cxr_info.explode("prev_dicom_id")
            print("Use only patient with at least one previous image: ", patient_cxr_info.shape)

            # Merge previous image information to the current image
            prev_cxr_meta = (
                cxr_meta[["dicom_id", "StudyDateTime", "StudyOrder"]].rename(columns={"dicom_id": "prev_dicom_id", "StudyDateTime": "prev_StudyDateTime", "StudyOrder": "prev_StudyOrder"}).copy()
            )
            patient_cxr_info = patient_cxr_info.merge(prev_cxr_meta, on="prev_dicom_id")
            patient_cxr_info = patient_cxr_info[
                (patient_cxr_info.prev_StudyDateTime < patient_cxr_info.StudyDateTime) & (patient_cxr_info.ADMITTIME < patient_cxr_info.prev_StudyDateTime)
            ]
            patient_cxr_info["StudyDateTime_diff"] = patient_cxr_info["StudyDateTime"] - patient_cxr_info["prev_StudyDateTime"]
            patient_cxr_info["StudyOrder_diff"] = patient_cxr_info["StudyOrder"] - patient_cxr_info["prev_StudyOrder"]
            patient_cxr_info = patient_cxr_info[patient_cxr_info.StudyOrder_diff > 0]
            patient_cxr_info = patient_cxr_info[patient_cxr_info.StudyDateTime_diff < np.timedelta64(48, "h")]

            # Get time interval between previous image and current image
            patient_cxr_info["INTIME"] = patient_cxr_info["prev_StudyDateTime"]
            patient_cxr_info["OUTTIME"] = (patient_cxr_info["StudyDateTime"] - patient_cxr_info["INTIME"]).dt.total_seconds() / 60
            patient_cxr_info["dicom_id"] = patient_cxr_info["dicom_id"] + "_" + patient_cxr_info["prev_dicom_id"]

            patient_cxr_info.to_csv(os.path.join(self.dest, f'{self.ehr_name}_cohort_meta.csv'), index=False)
            patient_cxr_info = patient_cxr_info.drop(columns=["prev_dicom_id", "prev_StudyDateTime", "prev_StudyOrder", "StudyDateTime", "StudyOrder", "ADMITTIME"])
            print(patient_cxr_info.shape)

        return patient_cxr_info

    def icd10toicd9(self, dx):
        gem = pd.read_csv(self.gem_path)
        dx_icd_10 = dx[dx["icd_version"] == 10]["icd_code"]

        unique_elem_no_map = set(dx_icd_10) - set(gem["icd10cm"])

        map_cms = dict(zip(gem["icd10cm"], gem["icd9cm"]))
        map_manual = dict.fromkeys(unique_elem_no_map, "NaN")

        for code_10 in map_manual:
            for i in range(len(code_10), 0, -1):
                tgt_10 = code_10[:i]
                if tgt_10 in gem["icd10cm"]:
                    tgt_9 = (
                        gem[gem["icd10cm"].str.contains(tgt_10)]["icd9cm"]
                        .mode()
                        .iloc[0]
                    )
                    map_manual[code_10] = tgt_9
                    break

        def icd_convert(icd_version, icd_code):
            if icd_version == 9:
                return icd_code

            elif icd_code in map_cms:
                return map_cms[icd_code]

            elif icd_code in map_manual:
                return map_manual[icd_code]
            else:
                logger.warn("WRONG CODE: " + icd_code)

        dx["icd_code_converted"] = dx.apply(
            lambda x: icd_convert(x["icd_version"], x["icd_code"]), axis=1
        )
        return dx


    def clinical_task(self, cohorts, task, spark):

        fname = self.task_itemids[task]["fname"]
        timestamp = self.task_itemids[task]["timestamp"]
        timeoffsetunit = self.task_itemids[task]["timeoffsetunit"]
        excludes = self.task_itemids[task]["exclude"]
        code = self.task_itemids[task]["code"][0]
        if "value" in self.task_itemids[task].keys():
            value = self.task_itemids[task]["value"][0]
        itemid = self.task_itemids[task]["itemid"]

        table = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
        table = table.drop(*excludes)
        if "value" in self.task_itemids[task].keys():
            table = table.filter(F.col(code).isin(itemid)).filter(F.col(value).isNotNull())
        else:
            table = table.filter(F.lower(F.col(code)).isin(itemid))
        
        merge = cohorts.join(table, on=self.hadm_key, how="inner")
        merge = merge.withColumn(timestamp, F.to_timestamp(timestamp))

        # Filter Dialysis at here to use abs timestamp & agg by patient_key
        # For Creatinine task, eliminate icus if patient went through dialysis treatment before (obs_size + pred_size / outtime) timestamp
        # Filtering base on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/treatment/rrt.sql (Dialysis Active)
        if task == "creatinine":
            dialysis_tables = self.task_itemids["dialysis"]["tables"]
            
            chartevents = spark.read.csv(os.path.join(self.data_dir, "icu/chartevents" + self.ext), header=True)
            inputevents = spark.read.csv(os.path.join(self.data_dir, "icu/inputevents" + self.ext), header=True)
            procedureevents = spark.read.csv(os.path.join(self.data_dir, "icu/procedureevents" + self.ext), header=True)
            
            chartevents = chartevents.select(*dialysis_tables["chartevents"]["include"])
            inputevents = inputevents.select(*dialysis_tables["inputevents"]["include"])
            procedureevents = procedureevents.select(*dialysis_tables["procedureevents"]["include"])

            # Filter dialysis related tables with dialysis condition #TODO: check dialysis condition
            ce = chartevents.filter((((F.col("itemid") == 225965) & (F.col("value") == "In use")) \
                | (F.col("itemid").isin(dialysis_tables["chartevents"]["itemid"]["ce"])) & F.col("value").isNotNull())
            )
            ie = inputevents.filter(F.col("itemid").isin(dialysis_tables["inputevents"]["itemid"]["ie"])).filter(F.col("amount") > 0)
            pe = procedureevents.filter(F.col("itemid").isin(dialysis_tables["procedureevents"]["itemid"]["pe"])).filter(F.col("value").isNotNull())

            # Extract Dialysis Times!
            def dialysis_time(table, timecolumn):
                return (table
                    .withColumn("_DIALYSIS_TIME", F.to_timestamp(timecolumn))
                    .select(self.patient_key, "_DIALYSIS_TIME")
                )
            ce, ie, pe = dialysis_time(ce, "charttime"), dialysis_time(ie, "starttime"), dialysis_time(pe, "starttime")
            dialysis = ce.union(ie).union(pe)
            dialysis = dialysis.groupby(self.patient_key).agg(F.min("_DIALYSIS_TIME").alias("_DIALYSIS_TIME"))
            merge = merge.join(dialysis, on=self.patient_key, how="left")

            # Only leave events with no dialysis / before first dialysis
            merge = merge.filter(F.isnull("_DIALYSIS_TIME") | (F.col("_DIALYSIS_TIME") > F.col(timestamp)))
            merge = merge.drop("_DIALYSIS_TIME")

        merge = (
            merge.withColumn(
                timestamp,
                F.round((F.col(timestamp).cast("long") - F.col("INTIME").cast("long")) / 60)
            )
        )

        # Cohort with events within (obs_size + gap_size) - (obs_size + pred_size / outtime)
        if self.rolling_from_last:
            merge = merge.filter(
                F.col(timestamp) <= F.col("OUTTIME") + self.pred_size * 60
            ).filter(F.col(timestamp)>= F.col("OUTTIME"))
        elif self.first_to_last:
            merge = merge.filter(
                F.col(timestamp) <= F.col("OUTTIME")
            ).filter(F.col(timestamp) >= F.col("OUTTIME") - self.pred_size * 60)
        else:
            merge = merge.filter(
                ((self.obs_size + self.gap_size) * 60) <= F.col(timestamp)
            ).filter(((self.obs_size + self.pred_size) * 60) >= F.col(timestamp))

        # Average value of events
        if "value" in self.task_itemids[task].keys():
            if self.first_to_last:
                window = Window.partitionBy(self.icustay_key).orderBy(F.desc(timestamp))
                value_agg = merge.withColumn("row", F.row_number().over(window)).filter(F.col("row") == 1).drop("row").withColumnRenamed(value, "avg_value")
            else:
                value_agg = merge.groupBy(self.icustay_key).agg(F.mean(value).alias("avg_value")) # TODO: mean/min/max?
        else:
            value_agg = merge.groupBy(self.icustay_key).agg(F.count(code).alias("event_count"))
            value_agg = (cohorts.select(self.icustay_key)
                         .join(value_agg.select(self.icustay_key, "event_count"), on=self.icustay_key, how="left")
                         .fillna(0, subset=["event_count"]))
        # Labeling
        if task == 'bilirubin':
            value_agg = value_agg.withColumn(task,
                F.when(value_agg.avg_value < 1.2, 0).when(
                    (value_agg.avg_value >= 1.2) & (value_agg.avg_value < 2.0), 1).when(
                        (value_agg.avg_value >= 2.0) & (value_agg.avg_value < 6.0), 2).when(
                            (value_agg.avg_value >= 6.0) & (value_agg.avg_value < 12.0), 3).when(
                                value_agg.avg_value >= 12.0, 4)
                )
        elif task == 'platelets':
            value_agg = value_agg.withColumn(task,
                F.when(value_agg.avg_value >= 150, 0).when(
                    (value_agg.avg_value >= 100) & (value_agg.avg_value < 150), 1).when(
                        (value_agg.avg_value >= 50) & (value_agg.avg_value < 100), 2).when(
                            (value_agg.avg_value >= 20) & (value_agg.avg_value < 50), 3).when(
                                value_agg.avg_value < 20, 4)
                )

        elif task == 'creatinine':
            value_agg = value_agg.withColumn(task,
                F.when(value_agg.avg_value < 1.2, 0).when(
                    (value_agg.avg_value >= 1.2) & (value_agg.avg_value < 2.0), 1).when(
                        (value_agg.avg_value >= 2.0) & (value_agg.avg_value < 3.5), 2).when(
                            (value_agg.avg_value >= 3.5) & (value_agg.avg_value < 5), 3).when(
                                value_agg.avg_value >= 5, 4)
                )

        elif task == 'wbc':
            # NOTE: unit is mg/L
            value_agg = value_agg.withColumn(task,
                F.when(value_agg.avg_value < 4, 0).when(
                    (value_agg.avg_value >= 4) & (value_agg.avg_value <= 12), 1).when(
                        (value_agg.avg_value > 12), 2)
                )
        
        elif task == 'hb':
            value_agg = value_agg.withColumn(task,
                F.when(value_agg.avg_value < 8, 0).when(
                    (value_agg.avg_value >= 8) & (value_agg.avg_value < 10), 1).when(
                        (value_agg.avg_value >= 10) & (value_agg.avg_value < 12), 2).when(
                            (value_agg.avg_value >= 12), 3)
                )

        elif task == 'bicarbonate':
            value_agg = value_agg.withColumn(task,
                F.when((value_agg.avg_value < 22), 0).when(
                        (value_agg.avg_value >= 22) & (value_agg.avg_value < 29), 1).when(
                            (value_agg.avg_value >= 29), 2)
            )

        elif task == 'sodium':
            value_agg = value_agg.withColumn(task,
                F.when(value_agg.avg_value < 135, 0).when(
                    (value_agg.avg_value >= 135) & (value_agg.avg_value < 145), 1).when(
                        (value_agg.avg_value >= 145), 2)
            )

        elif task == 'antibiotics':
            value_agg = value_agg.withColumn(task,
                F.when(value_agg.event_count < 1, 0).when(
                    (value_agg.event_count >= 1), 1)
            )

        cohorts = cohorts.join(value_agg.select(self.icustay_key, task), on=self.icustay_key, how="left")

        return cohorts
    
    
    def infer_data_extension(self) -> str:
        if (
            len(glob.glob(os.path.join(self.data_dir, "hosp", "*.csv.gz"))) == 21
            or len(glob.glob(os.path.join(self.data_dir, "icu", "*.csv.gz"))) == 8
        ):
            ext = ".csv.gz"
        elif (
            len(glob.glob(os.path.join(self.data_dir, "hosp", "*.csv")))==21
            or len(glob.glob(os.path.join(self.data_dir, "icu", "*.csv")))==8
        ):
            ext = ".csv"
        else:
            raise AssertionError(
                "Provided data directory is not correct. Please check if --data is correct. "
                "--data: {}".format(self.data_dir)
            )

        logger.info("Data extension is set to '{}'".format(ext))

        return ext