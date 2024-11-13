import os
from copy import deepcopy

import pandas as pd
import streamlit as st

from configs import configs
from src import add_background_labels
from utils import render_html, reset_state, get_ext


@st.cache_data(show_spinner=False)
def get_cbs_candidates(image_stats: pd.DataFrame, n: int=6):
    """Get n candidate images to be used as a CBS canvas"""
    image_filepaths = image_stats['overall']['filepath']
    return image_filepaths.sample(n).values.tolist()


def check_subfolders(cur_dir: str, subfolders: list) -> bool:
    """Check if subfolders exist in provided directory"""
    for subfolder in subfolders:
        if subfolder not in os.listdir(cur_dir):
            return False
    return True


def backup(x: str, session_state) -> None:
    """Callback to back up session state variables"""
    if isinstance(x, list):
        for k in x:
            session_state[f'{k}_backup'] = session_state[k]
        return
    session_state[f'{x}_backup'] = session_state[x]



def check_number_inputs(session_state) -> None:
    """Check if box size number inputs are correct"""
    defaults = {k:v for k, v in configs.PARAMS_DEFAULT_ANNOTATIONS.items()
                if k in ['medium_low', 'medium_high']}

    if session_state['medium_low'] > session_state['medium_high']:
        reset_state(
            session_state=session_state,
            defaults=defaults
        )
        st.toast(
            configs.TOAST_NUMBER_INPUTS,
            icon=":material/warning:")


def switch_toggle(attribute: str, session_state) -> None:
    """Callback to be used with toggles"""
    session_state[attribute] = not session_state[attribute]


def recover_toggle(session_state) -> None:
    """Recover toggle values when changing pages"""
    for attr_remove, attr_toggle in zip(configs.DATAMATCH_ATTRIBUTES_SHORT_REMOVE,
                                        configs.DATAMATCH_ATTRIBUTES_SHORT_TOGGLE):
        if session_state[attr_remove]:
            session_state[attr_toggle] = True

    if session_state['remove_lnl']:
        session_state['toggle_lnl'] = True

    if session_state['remove_mbg']:
        session_state['toggle_mbg'] = True

def remove_state(session_state, but: tuple=()) -> None:
    for key in session_state.keys():
        if not any(key.startswith(prefix) for prefix in but):
            del session_state[key]


def recover_second_page(session_state, same: bool=False) -> None:
    """Recover second page variables"""
    backups = configs.BACKUPS.copy()
    if same:
        del backups['stats_selectbox_backup']

    for backup_param in backups:
        if "_backup" in backup_param:
            if st.session_state[backup_param]:
                widget = backup_param.replace("_backup", "")
                if widget in session_state:
                    del session_state[widget]
                session_state[widget] = session_state[backup_param]


def toggle_disabled(attr: str, matching: dict, layout: dict) -> bool:
    """Callback to enable/disable toggles"""
    to_display_name = {'toggle_no_ext': 'No extension',
                       'toggle_wf': 'Wrong format',
                       'toggle_dp': 'Duplicates by filename'}

    if attr == 'toggle_lnl':
        if any(matching[f'Lonely {x}'] for x in configs.CATEGORIES):
            return False
        return True

    if attr == 'toggle_mbg':
        if matching[f'Lonely images']:
            return False
        return True

    if any(layout[x][to_display_name[attr]] for x in configs.CATEGORIES):
        return False
    return True


def render_disabled(session_state) -> bool:
    """Callback to enable/disable render button"""
    toggles = ['toggle_no_ext', 'toggle_wf',
               'toggle_dp', 'toggle_lnl']

    active_toggle = any([session_state[param] for param in toggles])
    files_pref = bool(session_state.files_pref)
    make_backgrounds =  session_state.toggle_mbg

    if (active_toggle and files_pref) or make_backgrounds:
        return False
    return True


def layout_check_with_toggle(session_state, layout: dict):
    """Update layout using toggles"""
    out = deepcopy(layout)
    for category in configs.CATEGORIES:
        for attribute, attribute_short_remove in zip(configs.DATAMATCH_ATTRIBUTES,
                                                     configs.DATAMATCH_ATTRIBUTES_SHORT_REMOVE):
            if session_state[attribute_short_remove]:
                if attribute == "Duplicates by filename":
                    out[category]['Correct'] = [i for i in out[category]['Correct']
                                                if i not in out[category][attribute]]
                out[category][attribute] = []

    return out


def matching_check_with_toggle(session_state, matching: dict, layout):
    """Update matching using toggles"""
    out1 = deepcopy(matching)
    out2 = deepcopy(layout)
    if session_state.remove_mbg:
        out2['labels']['Correct'] += add_background_labels(st.session_state.data_path,
                                                            layout, matching,
                                                            actual_fill=False)

        out1['Matched images'] += out1['Lonely images']
        out1['Matched labels'] += out1['Lonely images']
        out1['Backgrounds'] = out1['Lonely images']
        out1['Lonely images'] = []
    if st.session_state.remove_lnl:
        out2['images']['Correct'] = [i for i in out2['images']['Correct']
                                     if i not in out1['Lonely images']]
        out2['labels']['Correct'] = [i for i in out2['labels']['Correct']
                                     if i not in out1['Lonely labels']]

        out1['Lonely images'] = []
        out1['Lonely labels'] = []

    return out1, out2


def check_input_dir(session_state, input_dir: str) -> bool:
    """Check if provided directory is valid"""
    if not input_dir:
        return False
    if not os.path.isdir(input_dir):
        st.toast(configs.TOAST_NOT_A_DIRECTORY,
                 icon=":material/error:")
        return False
    else:
        structure_ok = check_subfolders(input_dir, configs.CATEGORIES)
        if structure_ok:
            session_state.data_path = input_dir.rstrip('/')
            return True
        else:
            st.toast(configs.TOAST_INVALID_DIRECTORY,
                     icon=":material/error:")
            return False


def check_input_page_toggles(page_name: str, session_state) -> None:
    """Callback to control toggles"""
    session_state[f'{page_name}_use_cached_backup'] = session_state[f'{page_name}_use_cached']
    session_state[f'{page_name}_cache_backup'] = session_state[f'{page_name}_cache']
    if session_state[f'{page_name}_use_cached']:
        session_state[f'{page_name}_sample_randomly'] = False
        session_state[f'{page_name}_cache'] = False
        session_state[f'{page_name}_forget_cached'] = False
    if session_state[f'{page_name}_cache']:
        session_state[f'{page_name}_forget_cached'] = False

def reset_input_page(page_name: str, session_state) -> None:
    """Callback to reset widgets"""
    session_state[f'{page_name}_use_cached_backup'] = session_state[f'{page_name}_use_cached'] = False
    session_state[f'{page_name}_cache_backup'] = session_state[f'{page_name}_cache'] = False
    session_state[f'{page_name}_sample_randomly'] = False
    session_state[f'{page_name}_cache'] = False
    session_state[f'{page_name}_forget_cached'] = False


def switch_board(label: str) -> None:
    """Callback to switch stats block"""
    st.session_state[f'show_first_board_{label}'] = not st.session_state[f'show_first_board_{label}']

def blank_line(n: int=1) -> None:
    """Add n blank lines"""
    st.markdown('<br>' * n, unsafe_allow_html=True)

def roll_planets(roll_configs: dict) -> None:
    """Planets animation"""
    for x, y, radius, size, cruise_time, rotation_time, key in zip(*roll_configs.values()):
        render_html(html_filepath='animations/planet.html',
                    uri=configs.URI[key],
                    x=x, y=y, radius=radius,
                    size=size, cruise_time=cruise_time,
                    rotation_time=rotation_time,key=key)

def add_layout_toggle(label: str, key: str, matching: dict, layout: dict,
                      session_state, hlp: str=None) -> None:
    """Add toggle for layout stats"""
    st.toggle(label=label, on_change=switch_toggle,key=key,
              disabled=toggle_disabled(key, matching, layout),
              kwargs=dict(attribute=key.replace('toggle', 'remove'),
                          session_state=session_state),
              help=hlp)



