import random
import shutil
from functools import partial
from pathlib import Path

from vis import plot_progress_pie
from .utils import *
from caching import get_cached_stats, cache_stats, forget_stats


@st.dialog("Confirmation")
def confirmation_dialog(arg: str, message: str, session_state) -> None:
    """Streamlit dialog to confirm an action"""
    st.markdown(message)

    blank_line()
    _, col1, col2 = st.columns([0.5, 0.25, 0.25])
    if col1.button('No', use_container_width=True):
        st.rerun()
    if col2.button('Yes', use_container_width=True):
        session_state[arg] = True
        st.rerun()


def box_sizes_table(session_state) -> None:
    """Create table for box high-low size thresholds"""
    medium_low = f"{round(session_state['medium_low'], 1)}%"
    medium_high = f"{round(session_state['medium_high'], 1)}%"

    table = pd.DataFrame(dict(
        low=['0%', medium_low, medium_high],
        high=[medium_low, medium_high, '100%']),
        index=['Small', 'Medium', 'Big']
    )

    st.table(table)


class StatsBlock:
    """
    A component for displaying statistics with toggleable views

    Args:
    label: Display label for the stats block
    metrics: Dictionary of metric names and their corresponding values
    n_samples: Number of samples to display (default: 20)
    width: Width of the stats block (default: 26)
    height: Height of the stats block (default: 37)
    """

    def __init__(
            self,
            label: str,
            metrics: dict,
            n_samples: int = 20,
            width: int = 26,
            height: int = 37
    ):

        self.label = label
        self.metrics = metrics
        self.n_samples = n_samples
        self.width = width
        self.height = height
        self.state_key = f'show_first_board_{label}'

        # Initialize state if not exists
        if self.state_key not in st.session_state:
            st.session_state[self.state_key] = True

    def render(self) -> None:
        """Render the complete stats block component."""
        self._render_styles()

        with st.form(key=self.label, clear_on_submit=False):
            self._render_header()

            if st.session_state[self.state_key]:
                self._render_metrics_view()
            else:
                self._render_samples_view()

    def _render_styles(self) -> None:
        """Render custom styles for the component."""
        render_html(
            'styles/stats_block.html',
            width=self.width,
            height=self.height
        )

    def _render_header(self) -> None:
        """Render the header section with toggle button and label."""
        col1, spacer, col2 = st.columns([20, 55, 25])

        with col1:
            st.form_submit_button(
                label='⇆',
                on_click=partial(switch_board, label=self.label)
            )

        with col2:
            with st.container(border=True):
                st.markdown(
                    f"<span style='font-size: 1vw; font-weight: bold; "
                    f"font-family: sans-serif'>{self.label}</span>",
                    unsafe_allow_html=True
                )

    def _render_metrics_view(self) -> None:
        """Render the metrics grid view."""
        row1, row2 = st.columns(2), st.columns(2)

        for col, metric_name in zip(row1 + row2, self.metrics.keys()):
            with col.container(border=True):
                st.metric(
                    metric_name,
                    len(self.metrics[metric_name])
                )

    def _render_samples_view(self) -> None:
        """Render the detailed samples view with tabs."""
        active_metrics = self._get_active_metrics()

        for tab, key in zip(st.tabs(active_metrics.keys()), active_metrics.keys()):
            with tab:
                self._render_sample_tab(key, active_metrics[key])

    def _get_active_metrics(self) -> dict:
        """Return metrics with non-empty values."""
        return {
            k: v for k, v in self.metrics.items()
            if len(v) > 0
        }

    def _render_sample_tab(self, key: str, samples: list) -> None:
        """
        Render individual sample tab content.

        Args:
            key: Metric key/name
            samples: List of samples for the metric
        """
        col1, col2 = st.columns([70, 30])

        with col1:
            self._render_sample_list(key, samples)

        with col2:
            self._render_sample_count(samples)

    def _render_sample_list(self, key: str, samples: list) -> None:
        """Render the list of samples with symbols."""
        symbol = '✅' if key == 'Correct' else '❌'
        selected_samples = random.sample(
            samples,
            min(len(samples), self.n_samples)
        )

        sample_entries = [
            f"{symbol}{Path(filename).name}"
            for filename in selected_samples
        ]

        separator = ("<hr style='width:100%;margin-bottom:0.3em;"
                     "margin-top:0.3em;border-width:2px'>")
        output_text = separator.join(sample_entries)

        st.markdown(
            f'<div class="scrollable-text">{output_text}</div>',
            unsafe_allow_html=True
        )

    @staticmethod
    def _render_sample_count(samples: list) -> None:
        """Render the count of samples."""
        st.markdown(
            f"<h1 style='text-align: center; color: black; "
            f"webkit-text-stroke-width: 0.1vw; "
            f"webkit-text-stroke-color: #FAFAFA;'>{len(samples)}</h1>",
            unsafe_allow_html=True
        )


def page_var(key: str, page_name: str, session_state):
    return session_state[f'{page_name}_{key}']


def caching_widgets(page_name: str,
                    total: int,
                    cached_exists: bool,
                    limit: int,
                    session_state):
    """
    Handles caching logic and UI controls for data processing pages.

    Args:
        page_name: Name of the current page
        total: Total number of records
        cached_exists: If cached version available
        limit: Number of instances to consider the data sufficient
        session_state: Streamlit session state dictionary

    Returns:
        Updated session state dictionary
    """

    input_page = st.empty()

    # Cache the partial function call to avoid repeated creation
    _page_var = partial(page_var,
                        page_name=page_name,
                        session_state=session_state)

    with input_page.container():
        # Create main layout with sidebar and content area
        col1, col2 = st.columns([0.15, 0.85])

        # Group all toggles together to minimize UI updates
        with col1:
            # === Sidebar Controls ===
            st.toggle(
                label='Use cached',
                key=f'{page_name}_use_cached',
                on_change=check_input_page_toggles,
                args=[page_name, session_state],
                disabled=not cached_exists)

            st.divider()

            is_sampling = st.toggle(
                label='Sample',
                value=False,
                key=f'{page_name}_sample_randomly',
                disabled=_page_var('use_cached'))

            sample_ratio = st.select_slider(
                label='Percentage',
                options=range(5, 101, 5),
                value=10,
                key=f'{page_name}_slider',
                disabled=not is_sampling)

            st.divider()

            # === Cache Management Controls ===
            st.toggle(
                label='Cache',
                key=f'{page_name}_cache',
                on_change=check_input_page_toggles,
                args=[page_name, session_state],
                disabled=_page_var('use_cached'))

            st.toggle(
                label='Forget cached',
                key=f'{page_name}_forget_cached',
                on_change=backup,
                args=[f'{page_name}_forget_cached', session_state],
                disabled=_page_var('use_cached') |  _page_var('cache'))

        # Pre-calculate colors once
        app_colored = _page_var('cache') or _page_var('use_cached')
        disk_colored = app_colored or _page_var('forget_cached')
        bin_colored = _page_var('forget_cached')

        # Batch render static UI elements
        icons = [
            ('app', configs.URI['app'], 52, 39, 3.5, 'App', app_colored),
            ('disk', configs.URI['disk'], 67, 36.5, 3.5, 'Disk', disk_colored),
            ('bin', configs.URI['bin'], 52.6, 57, 8, '∅', bin_colored)
        ]

        for key, uri, x, y, size, label, colored in icons:
            render_html(
                html_filepath='animations/slider.html',
                uri=uri,
                x=x,
                y=y,
                size=size,
                label=label,
                key=key,
                scale='colored' if colored else 'gray')

        # Render arrows only when needed
        arrows = []
        if  _page_var('use_cached'):
            arrows.append(('disk_to_app', 65, 42, 90))
        if  _page_var('cache'):
            arrows.append(('app_to_disk', 65, 41, -90))
        if  _page_var('forget_cached'):
            arrows.append(('cash_to_bin', 70, 52, 40))

        for key, x, y, direction in arrows:
            render_html(
                html_filepath='animations/arrows.html',
                x=x,
                y=y,
                size=0.6,
                direction=direction,
                key=key)

        sample_size = total if not is_sampling else int((sample_ratio / 100) * total)
        session_state[f'{page_name}_sample_size'] = sample_size

        disabled = sample_size < limit
        if disabled:
            st.toast(f'At least {limit} samples needed to proceed',
                     icon=':material/info:')

        # Add go button
        with col1:
            st.divider()

            if st.button(
                    label="Let's go",
                    use_container_width=True,
                    disabled=disabled,
                    key=f'{page_name}_go',
                    icon=':material/rocket_launch:'
            ):
                if not session_state[f'{page_name}_use_cached']:
                    session_state['processing'] = True
                session_state[f'{page_name}_input_page'] = False
                input_page.empty()
                st.rerun()


        # Calculate and display sample size
        with col2:
            blank_line(5)
            plot_progress_pie(
                progress=sample_size,
                total=total,
                label=page_name.capitalize(),
                color='#4f5052' if  _page_var('use_cached') else '#33B340')

    return session_state


def get_stats(page_name: str,
              getter_function: callable,
              spinner_message: str,
              session_state):

    """
    Collect and return stats using the getter function

    Args:
        page_name: Name of the current page
        getter_function: Function to use when collecting stats
        spinner_message: Message to display with spinner
        session_state: Streamlit session state dictionary

    Returns:
        Updated session state dictionary
    """

    def stop_processing():
        session_state[f'{page_name}_cancel'] = True

    # scan images and collect stats
    sample_size = session_state[f'{page_name}_sample_size']

    col1, col2 = st.columns([0.8, 0.2], gap='large')
    # stop image scanning process

    # if scanning images
    if session_state.processing:
        # create cancel button
        with col2:
            st.button(label='Cancel',
                      key=f'{page_name}_button',
                      on_click=stop_processing)

    with st.spinner(spinner_message):
        with col1:
            # get image stats
            stats = getter_function(sample_size=sample_size,
                                    cache_key=session_state[f'{page_name}_cache_key'])

            if session_state.processing:
                # set processing to False when it's over
                session_state.processing = False
                # None if the process was stopped
                if stats is None:
                    reset_input_page(page_name=page_name,
                                     session_state=session_state)
                    # clear cache on get_stats method
                    session_state[f'{page_name}_cache_key'] += 1
                    # return back to input_page
                    session_state[f'{page_name}_input_page'] = True

                st.rerun()

    # remove stats df from cash
    if session_state[f'{page_name}_forget_cached_backup']:
        forget_stats(
            path=f'.cache/{page_name}_stats.h5',
            key=session_state.data_path)
        session_state[f'{page_name}_forget_cached_backup'] = False

    # cache stats
    if session_state[f'{page_name}_cache_backup']:
        cache_stats(
            stats=stats,
            path=f'.cache/{page_name}_stats.h5',
            key=session_state.data_path)

        get_cached_stats.clear(path=f'.cache/{page_name}_stats.h5',
                               key=session_state.data_path)

        session_state[f'{page_name}_cache_backup'] = False
        session_state[f'{page_name}_use_cached'] = True
        session_state[f'{page_name}_use_cached_backup'] = True


    return stats, session_state

def make_toys():
    """Create coco128 toy datasets"""
    # Remove existing toys and cache
    if os.path.exists('toys'):
        if os.path.exists('.cache'):
            for cached_file in os.listdir('.cache'):
                if get_ext(cached_file) != 'h5':
                    continue
                for toy in os.listdir('toys'):
                    key = os.path.join(os.getcwd(), f'toys/{toy}')
                    forget_stats(path=f'.cache/{cached_file}', key=key)
        # Remove the directory and all its contents
        shutil.rmtree('toys')

    # Create a new 'toys' directory
    os.mkdir('toys')
    for toy in ['coco128-yolo', 'coco128-voc', 'coco128-mixed']:
        toy_type = toy.split('-')[1]
        os.mkdir(f'toys/{toy}')
        shutil.copytree(f'toy_data/{toy_type}_labels', f'toys/{toy}/labels')
        shutil.copytree('toy_data/images', f'toys/{toy}/images')

        if toy_type != 'voc':
            shutil.copy('toy_data/labelmap.txt', f'toys/{toy}/labelmap.txt')

