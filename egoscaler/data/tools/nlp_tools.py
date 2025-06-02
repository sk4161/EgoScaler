import re
from .suject_verb_object_extraction import findSVOs, nlp
import datetime

def time_str_to_sec(time_str):
    time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
    total_seconds = time_obj.second + time_obj.minute * 60 + time_obj.hour * 3600 + time_obj.microsecond / 1e6
    return total_seconds

def lemmatize_description(desc: str):
    
    desc = re.sub('#. |\t|\n', '', re.sub('  ', ' ', desc)).lower()            
    desc = re.sub('\.\.', '\.', desc)
    doc = nlp(desc)
    
    lemma_desc = " ".join([token.lemma_ for token in doc])
    
    return lemma_desc

def extract_verb_obj(desc: str):
    """
    narration: lammatized narration
    """
    _verb, _object = None, None
    desc = ' '.join(['I'] + desc.split(' ')[1:])
    tokens = nlp(desc)
    svos = findSVOs(tokens)
    if len(svos):
        svos = svos[0]
        if len(svos) == 3:
            _verb = svos[1]
            _object = re.sub('the |a |an ', '', svos[2])
    
    return _verb, _object

def which_hand(narr):
    hand_part = re.findall(r'with ((his|her)\s)?(left|right|both)?\s?hand', narr)
    if len(hand_part):
        hand_part = hand_part[0]
        if 'left' in hand_part:
            return 'left'
        elif 'right' in hand_part:
            return 'right'
        else:
            return None
    else:
        return None
    
def is_previous_action(narr):
    if re.findall('holds|moves|places', narr):
        return True
    else:
        return False

def format_tool(tool):
    """
    tool: raw llama3 output
    """
    tool = re.findall(r"\'.*\'", tool)
    if len(tool):
        tool = re.sub("\'", "", tool[0])
    else:
        tool = None
    return tool

def hand_transfer_flag(raw_desc):
    """
    Removes instances where an object is passed between hands,
    e.g., 'from his right hand to his left hand'.
    """
    # 正規表現で 'from X hand to Y hand' のパターンを検出
    pattern_transfer = r"\bfrom (the|his|her) (right|left|both) (hand|hands) to (the|his|her) (right|left|both) (hand|hands)\b"
    
    # パターンにマッチする場合、インスタンスを空文字にして削除
    if re.search(pattern_transfer, raw_desc, flags=re.IGNORECASE):
        return True  # 該当する場合は削除対象（Noneを返す）
    
    return False


def process_hand_mentions(raw_desc):
    """
    Processes 'hand(s)' mentions in raw_desc:
    1. If 'with the/his/her X in the/his/her (right|left|both) hand(s)', keeps 'with the/his/her X'.
    2. Removes 'with the/his/her X hand(s)' entirely.
    """
    # pattern 1: 'with the/his/her X in the/his/her (right|left|both) hand(s)' → 'with the/his/her X'
    pattern_case1 = r"\bwith (the|his|her) (\w+(?: \w+)?) in (the|his|her) (right|left|both) (hand|hands)\b"
    raw_desc = re.sub(pattern_case1, r"with \1 \2", raw_desc, flags=re.IGNORECASE)

    # pattern 2: 'with the/his/her X hand(s)' → 削除
    pattern_case2 = r"\bwith (the|his|her)(?: (\w+(?: \w+)?))? (hand|hands)\b"
    raw_desc = re.sub(pattern_case2, "", raw_desc, flags=re.IGNORECASE)

    # remove extra spaces
    raw_desc = re.sub(r'\s+', ' ', raw_desc).strip()
    return raw_desc

def format_description(desc: str) -> str:
    desc = desc.lstrip()
    desc = re.sub(r'\s+', ' ', desc)
    desc = re.sub(r'\.\s+', '.', desc)
    if not desc.endswith('.'):
        desc += '.'
    return desc