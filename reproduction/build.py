import pathlib
import re
import unicodedata
import math
import numpy as np
import pandas as pd
from source_io import fetch

MON_URL = 'https://data-donnees.az.ec.gc.ca/api/file?path=%2Fsubstances%2Fplaninfrastruture%2Fwastewater-systems-effluent-regulations-reported-data%2FResaeu-Wser-surveillance-monitoring.csv'
ID_URL = 'https://data-donnees.az.ec.gc.ca/api/file?path=%2Fsubstances%2Fplaninfrastruture%2Fwastewater-systems-effluent-regulations-reported-data%2FResaeu-Wser-identification.csv'
ACUTE_URL = 'https://data-donnees.az.ec.gc.ca/api/file?path=%2Fsubstances%2Fplaninfrastruture%2Fwastewater-systems-effluent-regulations-reported-data%2FResaeu-Wser-aigue-acute.csv'

TARGET = 'next_year_exceedance'
ID_COL = 'record_id'
TIME_COL = 'decision_date'
FEATURES = ['prior_year_exceedance', 'prior_year_worst_margin', 'ever_exceeded_before', 'history_years_reported', 'periods_reported_prior_year', 'reporting_frequency', 'system_type', 'avg_daily_effluent_volume_m3', 'owner_type_operator_type', 'province', 'treatment_family', 'hydraulic_retention_ge_5_days', 'chlorine_use', 'receiving_water_type', 'prior_year_acute_lethality_failures']
CARRY = ['target_year', 'system_id']
OUT_COLUMNS = [ID_COL, TIME_COL] + CARRY + FEATURES + [TARGET]
FIRST_LIMIT_YEAR = 2015
LAST_TARGET_YEAR = 2024

SYSTEM_ID_SYNONYMS = ['system id', 'system identifier', 'facility id', 'facility identifier', 'facility identification', 'identification de l installation', 'identificateur du systeme', 'identificateur de systeme', 'numero du systeme', 'no du systeme', 'system number', 'identifiant du systeme']
ACUTE_RESULT_SYNONYMS = ['acute lethality test result', 'test result', 'result', 'resultat', 'lethality', 'letality', 'mortality', 'mortalite', 'outcome']
MISSING_TOKENS = ('not provided', 'non fourni', 'not available', 'non disponible', 'na', 'n a', 'none', 'null', 'unknown', 'inconnu', 'not applicable', 'non applicable', 'no data', 'not reported', 'non renseigne', 'ne sais pas')


def _norm_text(value):
    if value is None:
        return ''
    text = unicodedata.normalize('NFKD', str(value))
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    text = re.sub('[^a-z0-9]+', ' ', text)
    return re.sub(' +', ' ', text).strip()


def _clean_id(value):
    if value is None:
        return ''
    text = str(value).strip()
    if text.endswith('.0') and text[:-2].isdigit():
        text = text[:-2]
    return text


def _to_num(value):
    if value is None:
        return np.nan
    try:
        if math.isnan(value):
            return np.nan
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return np.nan
    n = _norm_text(text)
    if n in MISSING_TOKENS:
        return np.nan
    text = text.replace(chr(160), ' ').replace(' ', '')
    text = text.replace('<', '').replace('>', '')
    if ',' in text and '.' in text:
        text = text.replace(',', '')
    elif ',' in text:
        text = text.replace(',', '.')
    text = re.sub('[^0-9eE+.-]', '', text)
    if text in ('', '-', '.'):
        return np.nan
    try:
        return float(text)
    except Exception:
        return np.nan


def _read_csv(path, **kwargs):
    for enc in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False, **kwargs)
        except UnicodeDecodeError:
            continue
    raise AssertionError('Could not decode CSV file ' + str(path))


def _fail_field(field, columns, file_label, extra):
    headers = '; '.join(str(c) for c in columns)
    msg = 'Field resolution failed for ' + str(field) + ' in ' + str(file_label) + '. Observed headers: ' + headers + '. Missing or ambiguous names. Selected archive member: not applicable, direct CSV. ' + str(extra)
    raise AssertionError(msg[:4000])


def _read_selected(path, mapping):
    header = _read_csv(path, nrows=0)
    cols = list(header.columns)
    seen = {}
    items = []
    for internal, source in mapping.items():
        if source is None:
            continue
        try:
            idx = cols.index(source)
        except ValueError:
            _fail_field(internal, cols, str(path), 'source header was not present in the selected CSV')
        if idx in seen:
            continue
        seen[idx] = internal
        items.append((idx, internal))
    items = sorted(items, key=lambda x: x[0])
    indices = [x[0] for x in items]
    names = [x[1] for x in items]
    if not indices:
        return pd.DataFrame(columns=list(mapping.keys()))
    for enc in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            parts = []
            reader = pd.read_csv(path, usecols=indices, dtype=str, keep_default_na=False, chunksize=200000, encoding=enc, header=0, low_memory=False)
            for chunk in reader:
                chunk.columns = names
                parts.append(chunk)
            if not parts:
                return pd.DataFrame(columns=names)
            return pd.concat(parts, ignore_index=True)
        except UnicodeDecodeError:
            continue
    raise AssertionError('Could not decode selected CSV file ' + str(path))


def _resolve(columns, field, phrases, file_label, all_of=None, exclude=None, required=True):
    best = None
    for idx, col in enumerate(columns):
        n = _norm_text(col)
        if not n:
            continue
        score = 0
        for phrase in phrases:
            p = _norm_text(phrase)
            if not p:
                continue
            if n == p:
                score = max(score, 1000 + len(p))
            elif p in n:
                score = max(score, 500 + len(p))
        for tokens in (all_of or []):
            if all(_norm_text(t) in n for t in tokens):
                score = max(score, 300 + sum(len(_norm_text(t)) for t in tokens))
        if exclude and any(_norm_text(x) in n for x in exclude):
            score = 0
        if score > 0:
            if best is None or score > best[0]:
                best = (score, idx, col)
    if best is None:
        if required:
            _fail_field(field, columns, file_label, 'No header matched the expected bilingual synonyms.')
        return None
    return best[2]


def _resolve_any(columns, field, phrase_groups, file_label, exclude=None, required=True):
    for phrases in phrase_groups:
        col = _resolve(columns, field, phrases, file_label, exclude=exclude, required=False)
        if col is not None:
            return col
    if required:
        _fail_field(field, columns, file_label, 'No header matched any synonym group: ' + str(phrase_groups))
    return None


def _find_all(columns, phrases, file_label, exclude=None):
    out = []
    for col in columns:
        n = _norm_text(col)
        if not n:
            continue
        if any(_norm_text(p) in n for p in phrases):
            if exclude and any(_norm_text(x) in n for x in exclude):
                continue
            out.append(col)
    if not out:
        _fail_field('required multi-column group', columns, file_label, 'No headers matched the expected treatment synonyms.')
    return out


def _parse_year(value):
    if value is None:
        return np.nan
    text = str(value)
    m = re.search('(19|20)[0-9][0-9]', text)
    if m:
        return int(m.group(0))
    return np.nan


def _period_token(period, month):
    n = _norm_text(period)
    if 'annual' in n or 'annuel' in n or 'annuelle' in n:
        return 'annual'
    m = re.search('q[ ]*([1-4])', n)
    if m:
        return 'Q' + m.group(1)
    m = re.search('quarter[ ]*([1-4])', n)
    if m:
        return 'Q' + m.group(1)
    m = re.search('trimestre[ ]*([1-4])', n)
    if m:
        return 'Q' + m.group(1)
    m = re.search('(?:19|20)[0-9]{2}[ ]?(0?[1-9]|1[0-2])$', n)
    if m:
        return 'M' + str(int(m.group(1)))
    mv = _to_num(month)
    if not math.isnan(mv):
        if int(mv) == 0:
            return 'annual'
        return 'M' + str(int(mv))
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    for i, name in enumerate(months, start=1):
        if name in n:
            return 'M' + str(i)
    french = ['janvier', 'fevrier', 'mars', 'avril', 'mai', 'juin', 'juillet', 'aout', 'septembre', 'octobre', 'novembre', 'decembre']
    for i, name in enumerate(french, start=1):
        if name in n:
            return 'M' + str(i)
    m = re.search('m[ ]*([0-9]{1,2})', n)
    if m:
        return 'M' + m.group(1)
    return 'unknown'


def _period_order(token):
    token = str(token)
    if token == 'annual':
        return 0
    if token.startswith('Q') and token[1:].isdigit():
        return int(token[1:])
    if token.startswith('M') and token[1:].isdigit():
        return int(token[1:])
    return 99


def _canon_frequency(value):
    n = _norm_text(value)
    if not n or n in MISSING_TOKENS:
        return ''
    if 'annual' in n or 'annuel' in n:
        return 'annual'
    if 'quarter' in n or 'trimest' in n:
        return 'quarterly'
    if 'month' in n or 'mensuel' in n:
        return 'monthly'
    return n


def _canon_system_type(value):
    n = _norm_text(value)
    if not n or n in MISSING_TOKENS:
        return ''
    if 'intermittent' in n or 'intermitt' in n:
        return 'intermittent'
    if 'continuous' in n or 'continu' in n:
        return 'continuous'
    return n


def _operational_flag(value):
    n = _norm_text(value)
    if not n:
        return np.nan
    if any(x in n for x in ['not operational', 'non operational', 'non operationnel', 'not in operation', 'closed', 'ferme', 'inactive', 'inactif', 'decommission', 'abandon']):
        return 0.0
    if any(x in n for x in ['operational', 'operationnel', 'in operation', 'en operation', 'active', 'actif', 'open', 'ouvert']):
        return 1.0
    return np.nan


def _to_bool01(value):
    n = _norm_text(value)
    if not n:
        return np.nan
    if n in MISSING_TOKENS:
        return np.nan
    if 'no chlor' in n or 'non chlor' in n or 'without chlor' in n or 'sans chlore' in n:
        return 0.0
    if n in ('yes', 'oui', 'true', '1', 'y', 't') or n.startswith('yes') or n.startswith('oui'):
        return 1.0
    if n in ('no', 'non', 'false', '0', 'n', 'f') or n.startswith('no ') or n.startswith('non '):
        return 0.0
    if 'chlorin' in n or 'chlore' in n:
        return 1.0
    return np.nan


def _classify_treatment(text):
    n = _norm_text(text)
    if not n or n in MISSING_TOKENS:
        return ''
    if any(x in n for x in ['no treatment', 'aucun traitement', 'none', 'not treatment', 'sans traitement']):
        return 'no_treatment'
    if any(x in n for x in ['activated sludge', 'boues activees', 'mechanical', 'mecanique', 'secondary', 'secondaire', 'tertiary', 'tertiaire', 'biofiltration', 'membrane', 'sbr', 'sequencing batch', 'aeration']):
        return 'mechanical'
    if any(x in n for x in ['lagoon', 'lagunage', 'etang', 'stabilization', 'stabilisation', 'waste stabilization']):
        return 'lagoon'
    return 'other'


def _hrt_ge5(value):
    raw = str(value)
    n = _norm_text(value)
    if not n:
        return np.nan
    if n in MISSING_TOKENS:
        return np.nan
    if n in ('yes', 'oui', 'y', 'true', '1', 't') or n.startswith('yes') or n.startswith('oui'):
        return 1.0
    if n in ('no', 'non', 'n', 'false', '0', 'f') or n.startswith('no ') or n.startswith('non '):
        return 0.0
    if '<' in raw and '=' not in raw:
        return 0.0
    if any(x in n for x in ['less than', 'under', 'moins de', 'inferieur']):
        return 0.0
    if any(x in n for x in ['or more', 'or greater', 'more than', 'greater than', 'at least', 'minimum', 'plus de', 'superieur', '5 days', 'cinq jours']):
        return 1.0
    nums = re.findall('[0-9]+(?:[.][0-9]+)?', n)
    if not nums:
        return np.nan
    try:
        val = float(nums[0])
    except Exception:
        return np.nan
    return 1.0 if val >= 5.0 else 0.0


def _infer_freq_tokens(tokens):
    ts = [str(t) for t in tokens]
    if any(t == 'annual' for t in ts):
        return 'annual'
    if any(t.startswith('Q') and t[1:].isdigit() for t in ts):
        return 'quarterly'
    if any(t.startswith('M') and t[1:].isdigit() for t in ts):
        return 'monthly'
    return ''


def _aggregate(df):
    if len(df) == 0:
        return pd.DataFrame()
    return df.groupby(['system_id', 'year'], dropna=False).agg(n_rows=('system_id', 'size'), n_samples=('sample_key', 'nunique'), has_pos_limit=('has_pos_limit', 'max'), has_eval=('has_eval', 'max'), any_exc=('any_exc', 'max'), any_unknown=('any_unknown', 'max'), worst_margin=('worst_margin', 'max'), avg_volume=('avg_daily_volume', 'mean'), reporting_frequency=('reporting_frequency', 'last'), system_type=('system_type', 'last'), owner_type=('owner_type', 'last'), operator_type=('operator_type', 'last'), province=('province', 'last')).reset_index()


def _combine_owner(owner, operator):
    def _one(value):
        if value is None:
            return ''
        text = str(value).strip()
        if not text:
            return ''
        if _norm_text(text) in MISSING_TOKENS:
            return ''
        return text
    o = _one(owner)
    p = _one(operator)
    if o and p:
        return o + ' | ' + p
    if o:
        return o
    if p:
        return p
    return ''


def _safe_int(value, default=0):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except Exception:
        return default


def _nonmissing_count(series):
    if series.dtype == object:
        return int(((series.notna()) & (series.astype(str).str.strip() != '')).sum())
    return int(series.notna().sum())


def _check(name, observed, expected, passed, kind=None):
    d = {'check': name, 'observed': str(observed), 'expected': str(expected), 'passed': bool(passed)}
    if kind is not None:
        d['kind'] = kind
    return d


def _assert_checks(checks):
    for c in checks:
        if not c['passed']:
            raise AssertionError(('Source check failed: ' + c['check'] + '. Observed: ' + c['observed'] + '. Expected: ' + c['expected'])[:4000])


def build(output: pathlib.Path, cache: pathlib.Path):
    output = pathlib.Path(output)
    cache = pathlib.Path(cache)
    (output / 'public').mkdir(parents=True, exist_ok=True)
    (output / 'private').mkdir(parents=True, exist_ok=True)

    mon_path = pathlib.Path(fetch(MON_URL, cache, 'wser_monitoring.csv'))
    id_path = pathlib.Path(fetch(ID_URL, cache, 'wser_identification.csv'))
    acute_path = pathlib.Path(fetch(ACUTE_URL, cache, 'wser_acute_lethality.csv'))

    mon_cols = list(_read_csv(mon_path, nrows=0).columns)
    id_cols = list(_read_csv(id_path, nrows=0).columns)
    acute_cols = list(_read_csv(acute_path, nrows=0).columns)

    mon_system = _resolve(mon_cols, 'system identifier', SYSTEM_ID_SYNONYMS, str(mon_path), exclude=['name', 'nom'])
    mon_period = _resolve(mon_cols, 'reporting period', ['reporting period', 'period reported', 'periode de declaration', 'periode visee', 'periode'], str(mon_path), exclude=['end', 'fin', 'start', 'debut'], required=False)
    mon_year = _resolve(mon_cols, 'reporting year', ['reporting year', 'calendar year', 'annee de declaration', 'annee civile', 'annee'], str(mon_path), exclude=['period', 'periode'], required=False)
    mon_month = _resolve(mon_cols, 'month', ['month', 'mois'], str(mon_path), required=False)
    if mon_period is None and mon_year is None:
        _fail_field('reporting period or reporting year', mon_cols, str(mon_path), 'Neither a period column nor a year column was found.')
    mon_cbod_avg = _resolve(mon_cols, 'CBOD average', ['cbod average', 'average cbod', 'cbod concentration', 'cbod value', 'average concentration cbod', 'dbo5 moyenne', 'moyenne dbo5', 'moyenne de dbo5', 'concentration dbo5', 'dbo5'], str(mon_path), exclude=['limit', 'limite', 'compliance', 'conformite', 'exceedance', 'depassement', 'percent', 'pourcentage'])
    mon_cbod_limit = _resolve(mon_cols, 'CBOD limit', ['cbod limit', 'limit cbod', 'cbod limits', 'limite dbo5', 'limite de dbo5', 'dbo5 limite', 'dbo5 limits'], str(mon_path), exclude=['compliance', 'conformite'])
    mon_ss_avg = _resolve(mon_cols, 'suspended solids average', ['suspended solids average', 'average suspended solids', 'average total suspended solids', 'total suspended solids', 'suspended solids concentration', 'tss', 'mes moyenne', 'moyenne mes', 'moyenne des matieres en suspension', 'matieres en suspension', 'solides en suspension'], str(mon_path), exclude=['limit', 'limite', 'compliance', 'conformite'])
    mon_ss_limit = _resolve(mon_cols, 'suspended solids limit', ['suspended solids limit', 'limit suspended solids', 'total suspended solids limit', 'limite des matieres en suspension', 'limite matieres en suspension', 'limite des solides en suspension', 'limite mes', 'mes limite'], str(mon_path), exclude=['compliance', 'conformite'])
    mon_system_type = _resolve(mon_cols, 'system type', ['system type', 'type of system', 'type de systeme', 'type du systeme'], str(mon_path))
    mon_freq = _resolve(mon_cols, 'reporting frequency', ['reporting frequency', 'frequency of reporting', 'frequence de declaration', 'frequence de report', 'frequence'], str(mon_path))
    mon_volume = _resolve(mon_cols, 'average daily effluent volume', ['average daily effluent volume', 'average daily volume', 'daily effluent volume', 'average daily flow', 'average flow', 'effluent volume', 'volume moyen d effluent', 'volume moyen de l effluent', 'debit moyen', 'debit d effluent', 'volume moyen'], str(mon_path))
    mon_owner = _resolve(mon_cols, 'owner type', ['owner type', 'type of owner', 'type de proprietaire', 'proprietaire'], str(mon_path))
    mon_operator = _resolve(mon_cols, 'operator type', ['operator type', 'type of operator', 'type d exploitant', 'exploitant'], str(mon_path))
    mon_province = _resolve(mon_cols, 'province', ['province', 'province or territory', 'province ou territoire'], str(mon_path))
    mon_status = _resolve(mon_cols, 'system status', ['system status', 'operational status', 'status of system', 'etat du systeme', 'etat de fonctionnement', 'statut du systeme'], str(mon_path), exclude=['approval', 'approbation'], required=False)

    mon_map = {'system_id': mon_system, 'period': mon_period, 'year': mon_year, 'month': mon_month, 'cbod_average': mon_cbod_avg, 'cbod_limit': mon_cbod_limit, 'suspended_solids_average': mon_ss_avg, 'suspended_solids_limit': mon_ss_limit, 'system_type': mon_system_type, 'reporting_frequency': mon_freq, 'avg_daily_volume': mon_volume, 'owner_type': mon_owner, 'operator_type': mon_operator, 'province': mon_province, 'status': mon_status}
    mon = _read_selected(mon_path, mon_map)
    for key in mon_map:
        if key not in mon.columns:
            mon[key] = ''

    mon['system_id'] = mon['system_id'].map(_clean_id)
    years = []
    tokens = []
    months = []
    for period, yearv, monthv in zip(mon['period'].tolist(), mon['year'].tolist(), mon['month'].tolist()):
        y = _parse_year(yearv)
        if math.isnan(y):
            y = _parse_year(period)
        tokens.append(_period_token(period, monthv))
        months.append(_to_num(monthv))
        years.append(y)
    mon['year'] = years
    mon['period_token'] = tokens
    mon['month_num'] = months
    mon = mon[mon['system_id'] != '']
    mon = mon[~mon['year'].isna()].copy()
    mon['year'] = mon['year'].astype(int)
    mon['period_order'] = mon['period_token'].map(_period_order)
    mon['sample_key'] = mon['year'].astype(str) + '-' + mon['period_token'].astype(str) + '-' + mon['month_num'].fillna(0).astype(int).astype(str)
    for col in ['cbod_average', 'cbod_limit', 'suspended_solids_average', 'suspended_solids_limit', 'avg_daily_volume']:
        mon[col] = mon[col].map(_to_num)
    mon['system_type'] = mon['system_type'].map(_canon_system_type)
    mon['reporting_frequency'] = mon['reporting_frequency'].map(_canon_frequency)
    mon['province'] = mon['province'].map(lambda x: '' if _norm_text(x) in MISSING_TOKENS else str(x).strip())
    mon['owner_type'] = mon['owner_type'].map(lambda x: '' if _norm_text(x) in MISSING_TOKENS else str(x).strip())
    mon['operator_type'] = mon['operator_type'].map(lambda x: '' if _norm_text(x) in MISSING_TOKENS else str(x).strip())
    mon['operational'] = mon['status'].map(_operational_flag)

    cbod_lim = mon['cbod_limit']
    cbod_avg = mon['cbod_average']
    ss_lim = mon['suspended_solids_limit']
    ss_avg = mon['suspended_solids_average']
    mon['cbod_pos_limit'] = cbod_lim > 0
    mon['cbod_eval'] = mon['cbod_pos_limit'] & cbod_avg.notna()
    mon['cbod_exc'] = mon['cbod_pos_limit'] & cbod_avg.notna() & (cbod_avg > cbod_lim)
    mon['cbod_unknown'] = mon['cbod_pos_limit'] & cbod_avg.isna()
    mon['cbod_margin'] = np.where(mon['cbod_eval'], cbod_avg - cbod_lim, np.nan)
    mon['ss_pos_limit'] = ss_lim > 0
    mon['ss_eval'] = mon['ss_pos_limit'] & ss_avg.notna()
    mon['ss_exc'] = mon['ss_pos_limit'] & ss_avg.notna() & (ss_avg > ss_lim)
    mon['ss_unknown'] = mon['ss_pos_limit'] & ss_avg.isna()
    mon['ss_margin'] = np.where(mon['ss_eval'], ss_avg - ss_lim, np.nan)
    mon['has_pos_limit'] = mon[['cbod_pos_limit', 'ss_pos_limit']].any(axis=1)
    mon['has_eval'] = mon[['cbod_eval', 'ss_eval']].any(axis=1)
    mon['any_exc'] = mon[['cbod_exc', 'ss_exc']].any(axis=1)
    mon['any_unknown'] = mon[['cbod_unknown', 'ss_unknown']].any(axis=1)
    mon['worst_margin'] = mon[['cbod_margin', 'ss_margin']].max(axis=1, skipna=True)

    pre_limit_mask = (mon['year'] < FIRST_LIMIT_YEAR) & ((mon['cbod_limit'] > 0) | (mon['suspended_solids_limit'] > 0))
    pre_limit_rows = int(pre_limit_mask.sum())
    pre_limit_systems = int(mon.loc[pre_limit_mask, 'system_id'].nunique())
    pre_years = sorted(int(x) for x in mon.loc[pre_limit_mask, 'year'].unique().tolist()) if pre_limit_rows else []
    pre_years_text = '; '.join(str(x) for x in pre_years) if pre_years else 'none'
    mon.loc[mon['year'] < FIRST_LIMIT_YEAR, ['cbod_limit', 'suspended_solids_limit']] = np.nan
    mon.loc[mon['year'] < FIRST_LIMIT_YEAR, ['cbod_pos_limit', 'ss_pos_limit', 'cbod_eval', 'ss_eval', 'cbod_exc', 'ss_exc', 'cbod_unknown', 'ss_unknown', 'has_pos_limit', 'has_eval', 'any_exc', 'any_unknown']] = False
    mon.loc[mon['year'] < FIRST_LIMIT_YEAR, ['cbod_margin', 'ss_margin', 'worst_margin']] = np.nan

    mon = mon.sort_values(['system_id', 'year', 'period_order', 'month_num'], kind='mergesort').reset_index(drop=True)

    inferred = mon.groupby(['system_id', 'year'], dropna=False)['period_token'].agg(lambda s: _infer_freq_tokens(s.tolist())).rename('reporting_frequency_inferred').reset_index()

    id_system = _resolve(id_cols, 'identification system identifier', SYSTEM_ID_SYNONYMS, str(id_path), exclude=['name', 'nom'])
    id_status = _resolve(id_cols, 'identification system status', ['system status', 'operational status', 'status of system', 'etat du systeme', 'etat de fonctionnement', 'statut du systeme'], str(id_path), exclude=['approval', 'approbation'], required=False)
    id_province = _resolve(id_cols, 'identification province', ['province', 'province or territory', 'province ou territoire'], str(id_path), required=False)
    treatment_cols = _find_all(id_cols, ['treatment type', 'type of treatment', 'treatment', 'traitement', 'type de traitement', 'disinfection', 'desinfection'], str(id_path), exclude=['name', 'nom', 'plant name'])
    hrt_col = _resolve(id_cols, 'hydraulic retention time', ['hydraulic retention time', 'retention time', 'temps de retention', 'temps de retention hydraulique', 'hrt'], str(id_path))
    chlorine_col = _resolve(id_cols, 'chlorine use', ['chlorine use', 'chlorination', 'chlorine', 'chlore', 'utilisation de chlore', 'desinfection au chlore'], str(id_path), exclude=['residual'])
    receiving_col = _resolve_any(id_cols, 'receiving water description at the discharge point', [
        ['description of water frequented by fish', 'description des eaux frequentes par les poissons', 'description des eaux frequentes par les poisons'],
        ['name of body of water frequented by fish', 'nom du plan d eau frequente par les poissons', 'nom du plan d eau frequente par les poisons'],
        ['water frequented by fish', 'eaux frequentes par les poissons', 'eaux frequentes par les poisons'],
        ['receiving water', 'receiving environment', 'receiving body', 'milieu recepteur', 'cours d eau recepteur', 'eau receptrice', 'description du milieu'],
    ], str(id_path), required=True)

    id_map = {'system_id': id_system}
    if id_status is not None:
        id_map['status_id'] = id_status
    if id_province is not None:
        id_map['province_id'] = id_province
    id_map['hrt_raw'] = hrt_col
    id_map['chlorine_raw'] = chlorine_col
    id_map['receiving_raw'] = receiving_col
    treatment_pairs = []
    for i, col in enumerate(treatment_cols):
        key = 'treatment_' + str(i)
        id_map[key] = col
        treatment_pairs.append((key, col))
    id_df = _read_selected(id_path, id_map)
    for key in ['status_id', 'province_id', 'hrt_raw', 'chlorine_raw', 'receiving_raw']:
        if key not in id_df.columns:
            id_df[key] = ''
    for key, header in treatment_pairs:
        if key not in id_df.columns:
            id_df[key] = ''
    id_df['system_id'] = id_df['system_id'].map(_clean_id)
    id_df = id_df[id_df['system_id'] != ''].copy()

    def _row_treatment(row, pairs):
        parts = []
        for key, header in pairs:
            v = row.get(key, '')
            n = _norm_text(v)
            if n in ('yes', 'oui', 'true', '1', 'y', 't'):
                parts.append(str(header))
            elif n in ('no', 'non', 'false', '0', 'n', 'f', 'not applicable', 'non applicable', 'na', 'n a'):
                continue
            elif str(v).strip():
                parts.append(str(v).strip())
        return ' | '.join(parts)

    def _clean_text(value):
        text = str(value).strip()
        n = _norm_text(text)
        if n in MISSING_TOKENS:
            return np.nan
        if not text:
            return np.nan
        return text

    id_df['treatment_text'] = id_df.apply(lambda row: _row_treatment(row, treatment_pairs), axis=1)
    id_df['treatment_family'] = id_df['treatment_text'].map(_classify_treatment)
    id_df['hydraulic_retention_ge_5_days'] = id_df['hrt_raw'].map(_hrt_ge5)
    id_df['chlorine_use'] = id_df['chlorine_raw'].map(_to_bool01)
    id_df['receiving_water_type'] = id_df['receiving_raw'].map(_clean_text)
    id_df['province_id'] = id_df['province_id'].map(lambda x: str(x).strip())
    id_df['status_id'] = id_df['status_id'].map(_operational_flag)
    id_dup_count = int(id_df['system_id'].duplicated().sum())
    id_df = id_df.sort_values('system_id').drop_duplicates('system_id', keep='last')
    id_by_system = id_df.set_index('system_id').to_dict('index')

    ac_system = _resolve(acute_cols, 'acute system identifier', SYSTEM_ID_SYNONYMS, str(acute_path), exclude=['name', 'nom'])
    ac_year = _resolve(acute_cols, 'acute test year or date', ['reporting year', 'year', 'annee', 'sample date', 'date d echantillonnage', 'test date', 'date du test', 'reporting period', 'periode', 'date'], str(acute_path))
    ac_result = _resolve(acute_cols, 'acute test result', ACUTE_RESULT_SYNONYMS, str(acute_path), exclude=['sample'])
    ac_species = _resolve(acute_cols, 'acute test species', ['species', 'espece', 'test species', 'espece testee'], str(acute_path), required=False)
    ac_map = {'system_id': ac_system, 'year_raw': ac_year, 'result': ac_result}
    if ac_species is not None:
        ac_map['species'] = ac_species
    ac = _read_selected(acute_path, ac_map)
    if 'species' not in ac.columns:
        ac['species'] = ''
    ac['system_id'] = ac['system_id'].map(_clean_id)
    ac = ac[ac['system_id'] != ''].copy()
    ac['year'] = ac['year_raw'].map(_parse_year)
    ac = ac[~ac['year'].isna()].copy()
    ac['year'] = ac['year'].astype(int)
    sp = ac['species'].map(_norm_text)
    sp_mask = sp.map(lambda x: any(y in x for y in ['rainbow', 'trout', 'truite', 'oncorhynchus']))
    if sp_mask.any():
        ac = ac[sp_mask].copy()

    def _is_fail(value):
        n = _norm_text(value)
        if not n:
            return False
        return ('fail' in n) or ('echec' in n) or ('mortality' in n and '50' in n)

    ac['failed'] = ac['result'].map(_is_fail)
    ac['result_known'] = ac['result'].map(lambda x: 1 if _norm_text(x) else 0)
    acute_counts = ac.groupby(['system_id', 'year'])['failed'].sum().to_dict()
    acute_known = ac.groupby(['system_id', 'year'])['result_known'].sum().to_dict()

    id_small = id_df[['system_id', 'province_id', 'status_id']].copy()
    target_all = mon[(mon['year'] >= FIRST_LIMIT_YEAR) & (mon['year'] <= LAST_TARGET_YEAR)].copy()
    target_all = target_all.merge(id_small, on='system_id', how='left')
    target_status_known = int(target_all['status_id'].notna().sum())
    current_status_nonoperational = int((target_all['status_id'] == 0).sum())
    excluded_rows = 0
    target = target_all.copy()

    all_agg = _aggregate(mon)
    all_agg = all_agg.merge(inferred, on=['system_id', 'year'], how='left')
    if 'reporting_frequency_inferred' not in all_agg.columns:
        all_agg['reporting_frequency_inferred'] = ''
    all_agg['reporting_frequency_inferred'] = all_agg['reporting_frequency_inferred'].fillna('')
    target_agg = _aggregate(target)
    pop = target_agg[target_agg['has_pos_limit'] & target_agg['has_eval']].copy()
    pop['label'] = pop['any_exc'].astype(int)
    pop['unresolved_no_positive'] = (~pop['any_exc']) & pop['any_unknown']
    pop = pop[~pop['unresolved_no_positive']].copy()
    all_by_system = {sid: sub.sort_values('year') for sid, sub in all_agg.groupby('system_id')}

    records = []
    feature_source_violations = 0
    prev_rows = 0
    freq_field_count = 0
    freq_inferred_count = 0
    for _, row in pop.iterrows():
        sid = row['system_id']
        y = int(row['year'])
        sub = all_by_system.get(sid)
        if sub is None:
            continue
        prior = sub[sub['year'] < y]
        history_years = int(prior['year'].nunique())
        if history_years < 1:
            continue
        if len(prior) and int(prior['year'].max()) >= y:
            feature_source_violations += 1
        prev = sub[sub['year'] == y - 1]
        prior_exc = np.nan
        prior_worst = np.nan
        periods_prior = 0
        freq = ''
        stype = ''
        vol = np.nan
        owner_op = ''
        province = ''
        if len(prev):
            prev_rows += 1
            p = prev.iloc[-1]
            if bool(p['has_eval']):
                prior_exc = float(int(p['any_exc']))
                wm = p['worst_margin']
                if pd.notna(wm):
                    prior_worst = max(float(wm), 0.0)
                else:
                    prior_worst = 0.0
            periods_prior = _safe_int(p['n_samples'], _safe_int(p['n_rows'], 0))
            raw_freq = p['reporting_frequency']
            freq = str(raw_freq) if pd.notna(raw_freq) else ''
            if freq.strip():
                freq_field_count += 1
            else:
                cand = p['reporting_frequency_inferred']
                freq = str(cand) if pd.notna(cand) else ''
                if freq.strip():
                    freq_inferred_count += 1
            stype = str(p['system_type']) if pd.notna(p['system_type']) else ''
            vol = float(p['avg_volume']) if pd.notna(p['avg_volume']) else np.nan
            owner_op = _combine_owner(p['owner_type'], p['operator_type'])
            province = str(p['province']) if pd.notna(p['province']) else ''
        eval_prior = prior[prior['has_eval']]
        if len(eval_prior):
            ever_exc = int(eval_prior['any_exc'].max())
        else:
            ever_exc = np.nan
        attrs = id_by_system.get(sid, {})
        if not province:
            province = str(attrs.get('province_id', '') or '')
        treatment_family = attrs.get('treatment_family', '')
        hrt = attrs.get('hydraulic_retention_ge_5_days', np.nan)
        chlor = attrs.get('chlorine_use', np.nan)
        receiving = attrs.get('receiving_water_type', np.nan)
        acute_val = np.nan
        key = (sid, y - 1)
        if key in acute_counts and acute_known.get(key, 0) > 0:
            acute_val = float(acute_counts[key])
        record = {ID_COL: 'wser-' + str(sid) + '-' + str(y), TIME_COL: str(y) + '-03-01', 'target_year': y, 'system_id': sid, 'prior_year_exceedance': prior_exc, 'prior_year_worst_margin': prior_worst, 'ever_exceeded_before': ever_exc, 'history_years_reported': history_years, 'periods_reported_prior_year': periods_prior, 'reporting_frequency': freq, 'system_type': stype, 'avg_daily_effluent_volume_m3': vol, 'owner_type_operator_type': owner_op, 'province': province, 'treatment_family': treatment_family, 'hydraulic_retention_ge_5_days': hrt, 'chlorine_use': chlor, 'receiving_water_type': receiving, 'prior_year_acute_lethality_failures': acute_val, TARGET: int(row['label'])}
        records.append(record)

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise AssertionError('No eligible system-year rows were constructed. Observed empty population. Expected at least one target year row with a positive limit and an evaluated average for a filing system.')
    for col in OUT_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    df = df[OUT_COLUMNS]
    df = df.sort_values([TIME_COL, 'system_id', ID_COL]).reset_index(drop=True)

    y2015 = df[df['target_year'] == FIRST_LIMIT_YEAR]
    y2015_prior_exc = int(y2015['prior_year_exceedance'].notna().sum())
    y2015_prior_margin = int(y2015['prior_year_worst_margin'].notna().sum())
    y2015_ever = int(y2015['ever_exceeded_before'].notna().sum())

    train = df[(df['target_year'] >= 2015) & (df['target_year'] <= 2019)].copy()
    eval_df = df[(df['target_year'] >= 2020) & (df['target_year'] <= 2022)].copy()
    holdout = df[(df['target_year'] >= 2023) & (df['target_year'] <= 2024)].copy()

    checks = []
    checks.append(_check('monitoring CSV parsed', str(len(mon)) + ' rows and ' + str(len(mon_cols)) + ' columns', 'nonempty monitoring CSV with source headers', len(mon) > 0))
    checks.append(_check('identification CSV parsed', str(len(id_df)) + ' unique systems and ' + str(len(id_cols)) + ' columns', 'nonempty identification CSV with one row per system', len(id_df) > 0))
    checks.append(_check('acute lethality CSV parsed', str(len(ac)) + ' rows and ' + str(len(acute_cols)) + ' columns', 'nonempty acute lethality CSV with source headers', len(ac) > 0))
    mon_required = [
        ('system identifier', mon_system),
        ('reporting period or reporting year', mon_period if mon_period is not None else mon_year),
        ('CBOD average', mon_cbod_avg),
        ('CBOD limit', mon_cbod_limit),
        ('suspended solids average', mon_ss_avg),
        ('suspended solids limit', mon_ss_limit),
        ('system type', mon_system_type),
        ('reporting frequency', mon_freq),
        ('average daily volume', mon_volume),
        ('owner type', mon_owner),
        ('operator type', mon_operator),
        ('province', mon_province),
    ]
    mon_missing = [name for name, col in mon_required if col is None]
    mon_resolved = [name + ' [' + str(col) + ']' for name, col in mon_required if col is not None]
    mon_observed = 'resolved: ' + '; '.join(mon_resolved) + ' | unresolved: ' + ('; '.join(mon_missing) if mon_missing else 'none') + ' | observed headers: ' + '; '.join(str(c) for c in mon_cols)
    checks.append(_check('required monitoring fields resolved', mon_observed, 'system identifier, at least one of reporting period or reporting year, CBOD average, CBOD limit, suspended solids average, suspended solids limit, system type, reporting frequency, average daily volume, owner type, operator type, province', len(mon_missing) == 0))
    checks.append(_check('required identification fields resolved', 'selected headers: ' + '; '.join([str(id_system), str(hrt_col), str(chlorine_col), str(receiving_col)]), 'identification system identifier, hydraulic retention answer, chlorine use indicator and water-frequented-by-fish description all resolved', all(x is not None for x in [id_system, hrt_col, chlorine_col, receiving_col])))
    checks.append(_check('identification system key is one row per facility', str(id_dup_count) + ' duplicate facility identifiers before deduplication; ' + str(len(id_df)) + ' facilities retained', 'identification file is a one-row-per-facility snapshot and deduplication leaves a unique key', len(id_df) > 0 and bool(id_df['system_id'].is_unique)))

    hrt_all = [v for v in id_df['hrt_raw'].map(_norm_text).tolist() if v]
    hrt_answers = [v for v in hrt_all if v not in MISSING_TOKENS]
    hrt_distinct = sorted(set(hrt_answers))[:8]
    hrt_mapped = int(id_df['hydraulic_retention_ge_5_days'].notna().sum())
    checks.append(_check('hydraulic retention field carries answered values', 'answer-like distinct values: ' + '; '.join(hrt_distinct) + '; mapped facilities ' + str(hrt_mapped) + ' of ' + str(len(id_df)), 'the identification column answers whether the hydraulic retention time is five days or more, so yes and no answers map to one and zero instead of being parsed as a numeric day count', (len(hrt_answers) == 0) or (hrt_mapped > 0)))

    receiving_mapped = _nonmissing_count(id_df['receiving_water_type'])
    receiving_norm = _norm_text(receiving_col)
    receiving_is_fish_water = ('water frequented by fish' in receiving_norm) or ('eaux frequentes' in receiving_norm)
    checks.append(_check('receiving-water type uses water-frequented-by-fish field', 'selected header: ' + str(receiving_col) + '; nonmissing receiving-water values: ' + str(receiving_mapped), 'selected header identifies the water frequented by fish, not the generic discharge place description, and at least one identification row carries a nonmissing value', receiving_is_fish_water and receiving_mapped > 0))

    join_cov = float(pop['system_id'].isin(id_df['system_id']).mean()) if len(pop) else 0.0
    checks.append(_check('identification join coverage', str(round(join_cov, 4)) + ' fraction of eligible system-years matched identification', 'at least 0.5 of eligible system-years match the identification snapshot', join_cov >= 0.5))

    status_year_aligned = any(('reporting year' in _norm_text(c)) or ('annee de declaration' in _norm_text(c)) for c in id_cols)
    checks.append(_check('operational status is year-aligned or not used for exclusion', 'year-aligned status field present ' + str(status_year_aligned) + '; target rows with known current-vintage status ' + str(target_status_known) + '; target rows with current-vintage non-operational status ' + str(current_status_nonoperational) + '; historical target rows excluded by current status ' + str(excluded_rows), 'a status used to exclude target years must carry a reporting-year stamp; without one, retain all filing systems and expose status only as a diagnostic limitation', status_year_aligned or excluded_rows == 0))

    checks.append(_check('target year range', 'minimum target year ' + str(int(df['target_year'].min())) + '; maximum target year ' + str(int(df['target_year'].max())), 'target years only 2015 through 2024', bool(df['target_year'].between(2015, 2024).all())))
    checks.append(_check('target population nonempty with positive limits', str(len(pop)) + ' eligible system-years; ' + str(int(pop['has_pos_limit'].sum())) + ' with a positive limit', 'nonempty population with at least one positive-limit row', len(pop) > 0))
    label_values = sorted(df[TARGET].unique().tolist())
    checks.append(_check('label is binary', '; '.join(str(x) for x in label_values), 'only 0 and 1', bool(set(label_values).issubset({0, 1}) and len(label_values) > 0)))
    unknown_zero = int((pop['any_unknown'] & (pop['label'] == 0)).sum())
    checks.append(_check('unknown outcomes excluded from negative labels', str(unknown_zero) + ' label-zero system-years still have an unresolved positive-limit average', 'zero; unresolved system-years are excluded rather than labeled 0', unknown_zero == 0))
    label_mismatch = int((pop['label'] != pop['any_exc'].astype(int)).sum())
    checks.append(_check('label follows strict average-minus-limit recomputation', str(label_mismatch) + ' mismatches between stored label and strict recomputation', 'zero; label is 1 only when a reported average is strictly greater than its positive limit', label_mismatch == 0))
    checks.append(_check('features use only prior target years', str(feature_source_violations) + ' cases where a feature source year was not earlier than target year', 'zero; historical features use rows with year less than target year', feature_source_violations == 0))
    checks.append(_check('reporting frequency recovered for prior-year aggregates', 'prior-year aggregates inspected ' + str(prev_rows) + '; frequency taken from the reported field ' + str(freq_field_count) + '; frequency recovered from prior-year period tokens ' + str(freq_inferred_count), 'reporting frequency comes from the latest prior-year monitoring row when reported and is inferred from prior-year period tokens when that field is blank, so the prior-year aggregate never needs period-token columns that only exist on unaggregated rows', (freq_inferred_count > 0) or (freq_field_count == prev_rows)))
    checks.append(_check('split row counts', 'train ' + str(len(train)) + '; eval ' + str(len(eval_df)) + '; holdout ' + str(len(holdout)), 'all three chronological splits are nonempty', len(train) > 0 and len(eval_df) > 0 and len(holdout) > 0))
    dup_count = int(df[ID_COL].duplicated().sum())
    checks.append(_check('stable record IDs are unique', str(dup_count) + ' duplicate record IDs', 'zero duplicate record IDs', dup_count == 0))
    checks.append(_check('output columns exactly declared', '; '.join(df.columns.tolist()), '; '.join(OUT_COLUMNS), bool(df.columns.tolist() == OUT_COLUMNS)))
    coverage = {}
    for f in FEATURES:
        coverage[f] = _nonmissing_count(df[f])
    zero_features = [f for f, c in coverage.items() if c == 0]
    checks.append(_check('declared feature coverage', '; '.join(f + ': ' + str(coverage[f]) for f in FEATURES), 'every declared feature has at least one nonmissing constructed value', len(zero_features) == 0))

    all_vals = pd.concat([mon['cbod_average'], mon['cbod_limit'], mon['suspended_solids_average'], mon['suspended_solids_limit']]).dropna()
    neg_count = int((all_vals < 0).sum()) if len(all_vals) else 0
    if len(all_vals):
        unit_obs = 'minimum ' + str(float(all_vals.min())) + '; maximum ' + str(float(all_vals.max())) + '; negative count ' + str(neg_count)
    else:
        unit_obs = 'no numeric concentration values'
    checks.append(_check('concentration measurement unit and frame', unit_obs, 'nonnegative concentrations in mg/L per the WSER description', len(all_vals) > 0 and neg_count == 0, kind='unit'))
    limit_vals = pd.concat([mon['cbod_limit'], mon['suspended_solids_limit']]).dropna()
    limit_vals = limit_vals[limit_vals > 0]
    count25 = int((limit_vals == 25.0).sum())
    checks.append(_check('default and facility-specific limit semantics', str(count25) + ' positive limit rows equal 25 mg/L; ' + str(len(limit_vals)) + ' positive limit rows total', 'the 25 mg/L default appears together with facility-specific transitional limits in the same columns', count25 > 0))
    checks.append(_check('pre-2015 source limit rows recorded; target labels begin 2015', str(pre_limit_rows) + ' source rows before 2015 carry a positive limit; systems ' + str(pre_limit_systems) + '; years ' + pre_years_text + '; minimum shipped target year ' + str(int(df['target_year'].min())), 'stray pre-2015 positive-limit rows are counted as evidence only; target labels begin at 2015; pre-2015 rows may still supply history counts but not limit-based features', int(df['target_year'].min()) >= FIRST_LIMIT_YEAR))
    id_has_reporting_year = any(('reporting year' in _norm_text(c)) or ('annee de declaration' in _norm_text(c)) for c in id_cols)
    checks.append(_check('identification joined as one snapshot row per facility', 'reporting-year header present ' + str(id_has_reporting_year) + '; unique facility identifiers after deduplication ' + str(int(id_df['system_id'].nunique())) + ' of ' + str(len(id_df)) + ' retained rows; current status excluded from historical row filtering', 'identification attributes are joined once per facility on the stable system identifier; with no reporting-year stamp those attributes stay current-vintage assumptions and current status is not used for historical exclusion', bool(id_df['system_id'].is_unique)))
    checks.append(_check('pre-2015 limit rows do not contribute to limit-based history', '2015 target rows ' + str(len(y2015)) + '; nonmissing prior_year_exceedance ' + str(y2015_prior_exc) + '; nonmissing prior_year_worst_margin ' + str(y2015_prior_margin) + '; nonmissing ever_exceeded_before ' + str(y2015_ever), 'zero; no limits applied before 2015, so limit-based historical features are missing for 2015 targets even when stray pre-2015 positive-limit rows exist', y2015_prior_exc == 0 and y2015_prior_margin == 0 and y2015_ever == 0))

    _assert_checks(checks)

    train.to_csv(output / 'public' / 'train.csv', index=False)
    eval_df.to_csv(output / 'public' / 'eval.csv', index=False)
    holdout.to_csv(output / 'private' / 'holdout.csv', index=False)
    return checks
