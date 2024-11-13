import subprocess
from copy import copy
import sys

from streamlit_extras.stylable_container import stylable_container
from streamlit_image_select import image_select

from front.elements import *
from vis import *
from caching import *
from configs import configs
from loaders.parsers import get_labelmap
from src import DataMatch, Annotations, Images, Overlaps
from src import render_layout, add_background_labels
from utils import double_callback, most_common, get_colormap
from utils import reset_state, set_state_for, PageScroll, render_html

import streamlit as st
import warnings

warnings.filterwarnings('ignore')

# set app layout to wide
st.set_page_config(
    page_title='Deeva - Dive deep into your data',
    page_icon='🔭',
    layout='wide'
)

# set the page scroller
set_state_for(key='page', value=PageScroll(0, 5), session_state=st.session_state)

# render styles for app appearance
render_html(html_filepath='styles/custom.html')
render_html(html_filepath='styles/background.html', uri=configs.URI['main_background'])

# get partials by freezing session attributes
_backup = partial(backup, session_state=st.session_state)
_recover_second_page = partial(recover_second_page, session_state=st.session_state)
_reset_state = partial(reset_state, session_state=st.session_state, defaults=configs.PARAMS_DEFAULT)

# input page
if st.session_state.page.current == 0:
    remove_state(
        session_state=st.session_state,
        but=('page', 'data_path', 'FormSubmitter:Question-Submit', 'recent', 'toys', 'reset_toys'))

    _reset_state(but=('reset_toys',))
    st.cache_data.clear()

    st.title('Ready for insights?')
    blank_line()

    # data path input
    col, _, col_examples = st.columns([0.25, 0.5, 0.25])
    with col:
        data_path = None
        for arg in sys.argv:
            if arg.startswith("--data_path="):
                data_path = arg.split("=")[1]
                sys.argv.remove(arg)

        if not data_path:
            with st.form("Question", clear_on_submit=False):
                data_path = st.text_input('Drop in the data path',
                                          placeholder='path/to/data',
                                          autocomplete='off',
                                          help=configs.HELP_DATA_PATH)
                submitted = st.form_submit_button()

            if submitted:
                if check_input_dir(st.session_state, data_path):
                    st.session_state.page.next()  # go to the next page
                    remember_path(st.session_state.data_path)

                    st.rerun()  # rerun to update session vars

        else:
            if check_input_dir(st.session_state, data_path):
                st.session_state.page.next()  # go to the next page
                remember_path(st.session_state.data_path)

                st.rerun()  # rerun to update session vars

    # recent paths
    saved_data_paths = get_saved_paths()

    col, _ = st.columns([0.225, 0.775])

    if saved_data_paths:
        with col.expander('Recent'):
            for saved_data_path in saved_data_paths:
                # links as invisible buttons
                with stylable_container(
                        key="invisible_button",
                        css_styles=open('styles/invisible_button.html', 'r').read()):

                    # path short form
                    max_length = 40
                    short_form = f'.../{os.path.basename(saved_data_path)}'
                    short_form = short_form[:max_length]

                    col1, col2 = st.columns([0.93, 0.07])

                    # if link clicked
                    if col1.button(short_form, key=f"recent_{saved_data_path}"):
                        st.session_state.data_path = saved_data_path
                        st.session_state.page.next()
                        remember_path(st.session_state.data_path)
                        st.rerun()

                    # if x(delete) clicked
                    if col2.button('✖', key=f"recent_{saved_data_path}_X"):
                        forget_path(saved_data_path)
                        st.rerun()

    col1, col2 = st.columns([0.2, 0.8])

    # create toys using toy data
    if not os.path.exists('toys'):
        make_toys()
    toy_datasets = os.listdir('toys')
    with col1.expander('Toys'):
        for dataset in toy_datasets:
            dataset_path = os.path.join(os.getcwd(), 'toys', dataset)
            # links as invisible buttons
            with stylable_container(
                    key="invisible_button",

                    css_styles=open('styles/invisible_button.html', 'r').read()):

                # if link clicked
                if st.button(dataset, key=f"toys_{dataset}"):
                    st.session_state.data_path = dataset_path
                    st.session_state.page.next()
                    st.rerun()

    if col2.button(':material/refresh:'):
        confirmation_dialog(arg='reset_toys',
                            message=configs.CONFIRMATION_RESET_TOYS,
                            session_state=st.session_state)

    if st.session_state.reset_toys:
        make_toys()
        st.toast(configs.TOAST_RESET_TOYS, icon=':material/info:')
        st.session_state.reset_toys = False

    # planets animation
    roll_planets(configs.ROLL_CONFIGS)

# datamatch page
elif st.session_state.page.current == 1:
    _recover_second_page()

    with st.spinner(text="Fetching stats"):
        data_match = DataMatch(st.session_state.data_path)

        # file categorization statistics
        layout = data_match.get_layout()
        layout_display = layout_check_with_toggle(session_state=st.session_state,
                                                  layout=layout)

        # corrupted files
        garbage = data_match.get_garbage()

        # data match statistics
        matching = data_match.get_matching(garbage)

        # live display(matched with toggles)
        matching_display, layout_display = matching_check_with_toggle(session_state=st.session_state,
                                                                      matching=matching,
                                                                      layout=layout_display)

    # get partials freezing session attributes
    _add_layout_toggle = partial(add_layout_toggle,
                                 matching=matching,
                                 layout=layout,
                                 session_state=st.session_state)

    # categorized statistics
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        images_block = StatsBlock(label='Images', metrics=layout_display['images'])
        images_block.render()

    with col2:
        labels_block = StatsBlock(label='Labels', metrics=layout_display['labels'])
        labels_block.render()

    col1, col2, col3 = st.columns([0.25, 0.45, 0.3])
    # supply arrow 1
    with col1:
        render_html(html_filepath='animations/supply.html',
                    d1=configs.D1[0],
                    d2=configs.D1[1],
                    color="#c0c0c0",
                    right=0,
                    key='left')

    # data matching
    with col2:
        vis_matching = VisMatching(matching=matching_display)
        vis_matching.plot()

    # supply arrow 2
    with col3:
        render_html(html_filepath='animations/supply.html',
                    d1=configs.D2[0],
                    d2=configs.D2[1],
                    color="#c0c0c0",
                    right=3,
                    key='right')

        # move to the next page
        _, col = st.columns([0.8, 0.2])
        if col.button(':material/arrow_forward:',
                      use_container_width=True):
            # to still be able to access stats from session_state
            st.session_state.pass_forward = {'layout': layout_display,
                                             'matching': matching_display}

            st.session_state.page.next()
            st.rerun()

    with st.sidebar:
        col1, col2, col3, _ = st.columns([0.15, 0.15, 0.15, 0.65], gap='medium')

        # move to the previous page
        if col1.button(':material/arrow_back:'):
            st.session_state.page.previous()
            st.rerun()

        # refresh
        if col2.button(':material/refresh:'):
            st.cache_data.clear()
            _reset_state(but=['data_path', 'page'])
            st.rerun()

        # open data path
        if col3.button(':material/folder_open:'):
            subprocess.Popen(['xdg-open', st.session_state.data_path])

        st.header('Layout', divider='grey')
        blank_line()

        # remove toggles
        _add_layout_toggle(label='Remove no-extension', key='toggle_no_ext')
        _add_layout_toggle(label='Remove wrong format', key='toggle_wf')
        _add_layout_toggle(label='Remove duplicates', key='toggle_dp')

        blank_line()
        st.header('Matching', divider='grey')
        blank_line()

        _add_layout_toggle(label='Remove lonely files', key='toggle_lnl')
        _add_layout_toggle(label='Mark all as backgrounds',
                           key='toggle_mbg',
                           hlp=configs.HELP_MARK_BACKGROUNDS)

        st.divider()

        st.title('Rendering')
        blank_line()

        # separate vs delete
        st.session_state.files_pref = st.radio(label="Rendering options",
                                               options=["Separate", "Delete"],
                                               captions=configs.CAPTIONS_FILES_PREF,
                                               label_visibility='collapsed',
                                               index=None,
                                               key='choose_fp')

        blank_line()
        _, col = st.columns([0.4, 0.6])
        render = col.button('Render and refresh',
                            disabled=render_disabled(st.session_state))

    if render:
        confirmation_dialog(arg='render_final',
                            message=configs.CONFIRMATION_WRITE_SOURCE,
                            session_state=st.session_state)

    # render layout
    if st.session_state.render_final:
        delete = True if st.session_state.files_pref == "Delete" else False
        render_layout(st.session_state, layout, matching, delete=delete)

        if st.session_state.remove_mbg:
            add_background_labels(data_path=st.session_state.data_path,
                                  layout=layout,
                                  matching=matching)

        st.session_state.render_final = False
        _reset_state(but=['data_path', 'page'])
        st.cache_data.clear()
        st.rerun()



# main stats page
elif st.session_state.page.current == 2:
    # to remember toggle configuration when changing pages
    recover_toggle(session_state=st.session_state)

    with st.spinner(text="Fetching stats"):
        correct_labels = st.session_state.pass_forward['layout']['labels']['Correct']
        correct_images = st.session_state.pass_forward['layout']['images']['Correct']
        matched_labels = st.session_state.pass_forward['matching']['Matched labels']
        matched_images = st.session_state.pass_forward['matching']['Matched images']

        # get labelmap
        labelmap, labelmap_warning = get_labelmap(st.session_state.data_path)

        annotations = Annotations(data_path=st.session_state.data_path,
                                  labels=correct_labels,
                                  labelmap=labelmap)

        # YOLO vs VOC vs corrupted stats
        annotation_formats = annotations.categorize_by_formats()
        # main annotation stats
        annotation_stats = annotations.get_stats()

        # match image filenames with label counterparts
        matching_dict = {k: v for k, v in zip(matched_labels, matched_images)}

        images = Images(data_path=st.session_state.data_path,
                        images=correct_images,
                        annotation_stats=annotation_stats)

        # jpg vs png vs jpeg vs corrupted stats
        image_format_counts = images.get_format_counts()


        overlaps = Overlaps(annotation_stats=annotation_stats,
                            matching_dict=matching_dict,
                            data_path=st.session_state.data_path)

        # object classes
        classes = sorted(annotation_stats['class_name'].unique().tolist())
        # colormap
        colormap = get_colormap(classes, configs.CLASS_COLORS)

        colormap['All'] = '#FFFFFF'  # white for "All" option
        classes.insert(0, 'All')

    with st.sidebar:
        col1, col2, col3, _ = st.columns([0.15, 0.15, 0.2, 0.5])
        # go back
        if col1.button(':material/arrow_back:',
                       disabled=st.session_state.processing):
            st.session_state.page.previous()
            st.rerun()

        # refresh
        if col2.button(':material/refresh:',
                       disabled=st.session_state.processing):
            _reset_state(but=['data_path', 'page', 'pass_forward'])
            st.cache_data.clear()
            st.session_state.page.previous()
            st.rerun()

        # root dir
        if col3.button(':material/folder_open:',
                       disabled=st.session_state.processing):
            subprocess.Popen(['xdg-open', st.session_state.data_path])

        # selectbox to choose between different statistics categories
        st.divider()
        st.selectbox(label='Select the stats',
                     options=['Overall', 'Annotations', 'Images', 'Overlaps'],
                     key='stats_selectbox',
                     label_visibility='collapsed',
                     on_change=double_callback,
                     args=(partial(_backup, x='stats_selectbox'),
                           partial(_recover_second_page, same=True)),
                     disabled=st.session_state.processing,
                     index=0)
        st.divider()

        stats_page = st.session_state.stats_selectbox

    # Overall page
    if stats_page == 'Overall':
        n_classes = annotation_stats['class_name'].nunique()

        # if literally no data to show
        if not correct_images or not correct_labels or not n_classes:
            # not enough data animation
            render_html(
                html_filepath='animations/astronaut.html',
                uri=configs.URI['astronaut'])

            st.stop()

        with st.sidebar:
            set_state_for(key='max_classes',
                          value=min(n_classes, 10),
                          session_state=st.session_state)
            set_state_for(key='scarce_threshold',
                          value=0.1,
                          session_state=st.session_state)

            col1, col2 = st.columns([0.55, 0.45], gap='medium')
            with col2:
                blank_line()
                st.checkbox(
                    label='Show all',
                    key='show_all_checkbox',
                    on_change=_backup,
                    kwargs=dict(x='show_all_checkbox'),
                )

            col1.number_input(
                label='Show classes',
                key='max_classes',
                min_value=1,
                max_value=n_classes,
                placeholder='max',
                on_change=_backup,
                kwargs=dict(x='max_classes'),
                disabled=st.session_state.show_all_checkbox,
                help=configs.HELP_MAX_CLASSES
            )

            blank_line()

            col1, col2 = st.columns([0.47, 0.53], gap='small')

            col1.number_input(
                label='Scarcity ratio',
                key='scarce_threshold',
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                placeholder='threshold',
                on_change=_backup,
                kwargs=dict(x='scarce_threshold'),
                format='%.2g',
                help=configs.HELP_SCARCE_THRESHOLD
            )

            col2.radio(
                label='Threshold type',
                options=['Max', 'Total'],
                key='threshold_type',
                horizontal=True,
                on_change=_backup,
                index=0,
                kwargs=dict(x='threshold_type'),
                label_visibility='hidden')

            st.divider()
            blank_line()

            # convert annotations
            with st.form('Convert annotations'):
                st.radio(label='Convert',
                         options=['To VOC', 'To YOLO'],
                         key='convert_radio',
                         index=None,
                         label_visibility='collapsed',
                         captions=configs.CAPTIONS_CONVERT)

                _, col = st.columns([0.55, 0.45])
                convert = col.form_submit_button('Convert', icon=':material/conversion_path:')

            # if clicked convert
            if convert:
                if not st.session_state.convert_radio:
                    st.toast(configs.TOAST_CONVERT, icon=':material/error:')
                else:
                    message = configs.CONFIRMATION_WRITE_SOURCE

                    # lonely yolo labels cannot be converted to voc
                    lonely_labels = st.session_state.pass_forward['matching']['Lonely labels']
                    if lonely_labels and st.session_state.convert_radio == 'To VOC':
                        message += configs.CONFIRMATION_LONELY_FILES.format(len(lonely_labels))

                    confirmation_dialog(arg='convert_final',
                                        message=message,
                                        session_state=st.session_state)

            # if confirmation dialogue exited positively
            if st.session_state.convert_final:
                # bring to destination format
                convert_option = st.session_state.convert_radio
                destination_format = convert_option.split(' ')[-1].lower()

                annotations.bring_to(annotation_format_stats=annotation_formats,
                                     destination_format=destination_format,
                                     matching_dict=matching_dict)

                st.cache_data.clear()
                st.session_state.page.previous()
                st.session_state.convert_final = False
                st.rerun()

        # annotations pie values
        n_yolo = len(annotation_formats['yolo'])
        n_voc = len(annotation_formats['voc'])
        n_corrupted_annotations = len(annotation_formats['corrupted'])

        # backgrounds pie values
        n_images = len(correct_images)
        n_backgrounds = len(annotation_formats['background'])

        # if created labels for lonely images
        if st.session_state.remove_mbg:
            # count lonely images too as backgrounds
            n_backgrounds += len(st.session_state.pass_forward['matching']['Backgrounds'])

        col1, col2 = st.columns([0.45, 0.55])
        # General overview
        with col1:
            plot_general_overview(
                annotations_counts=[n_yolo, n_voc, n_corrupted_annotations],
                backgrounds_counts=[n_images - n_backgrounds, n_backgrounds],
                images_counts=list(image_format_counts.values())
            )

        with col2:
            # class and image counts bar chart
            max_classes = n_classes if st.session_state.show_all_checkbox else st.session_state.max_classes
            scarce_threshold = st.session_state.scarce_threshold
            threshold_type = st.session_state.threshold_type.lower()

            vis_classes = VisClasses(annotation_stats=annotation_stats,
                                     max_classes=max_classes,
                                     threshold=scarce_threshold,
                                     threshold_type=threshold_type)

            relevant_classes = vis_classes.plot()

            n_classes = annotation_stats['class_name'].nunique()
            if n_classes > max_classes:
                st.toast(
                    body=f"Only **{max_classes}/{n_classes}** classes displayed for visual interpretability",
                    icon=':material/notification_important:')

            # object class co-occurrence matrix
            vis_co_occurrences = VisCoOccurrences(annotation_stats=annotation_stats,
                                                  relevant_classes=relevant_classes)
            vis_co_occurrences.plot()

        # toast a warning if labelmap was not found
        if labelmap_warning  and n_yolo:
            toast = st.toast(labelmap_warning, icon=":material/warning:")
            if n_voc:
                st.toast(configs.TOAST_MIXED_FORMATS,
                         icon=":material/warning:")

    # Annotations page
    if stats_page == 'Annotations':
        n_annotations = annotation_stats.shape[0]
        if n_annotations < configs.ANNOTATIONS_LIMIT:
            # not enough data animation
            render_html(
                html_filepath='animations/astronaut.html',
                uri=configs.URI['astronaut'])

            st.stop()

        with st.sidebar:
            # setting default parameters for annotations widgets
            for key, value in configs.PARAMS_DEFAULT_ANNOTATIONS.items():
                set_state_for(key=key,
                              value=value,
                              session_state=st.session_state)

            col1, col2 = st.columns(2)

            # object class selector
            col1.selectbox(label='Class',
                           options=classes,
                           key='annotations_class_selectbox',
                           on_change=_backup,
                           kwargs=dict(x='annotations_class_selectbox'),
                           index=0)

            # plot type selector
            col2.selectbox(label='Plot type',
                           options=['Scatter', 'Heatmap'],
                           key='plot_type_selectbox',
                           on_change=_backup,
                           kwargs=dict(x='plot_type_selectbox'),
                           index=0)

            blank_line()

            # annotation controls
            with st.form('Annotation controls'):
                blank_line()

                # check if high >= low
                check_number_inputs(session_state=st.session_state)

                col1, col2 = st.columns(2)
                # low threshold for medium size
                col1.number_input(label='Medium low',
                                  min_value=0.0,
                                  max_value=100.0,
                                  step=0.1,
                                  placeholder='low',
                                  key='medium_low',
                                  format='%.1f')

                # high threshold for medium size
                col2.number_input(label='Medium high',
                                  min_value=0.0,
                                  max_value=100.0,
                                  step=0.1,
                                  placeholder='high',
                                  key='medium_high',
                                  format='%.1f')

                # box size range selector
                st.slider(label='Box size',
                          min_value=0.0,
                          max_value=100.0,
                          step=0.1,
                          key='box_size_slider',
                          format='%.1f%%')

                _, col = st.columns([0.6, 0.4])

                col.form_submit_button(label='Apply',
                                       use_container_width=True,
                                       on_click=_backup,
                                       kwargs=dict(
                                           x=['medium_low',
                                              'medium_high',
                                              'box_size_slider']),
                                       icon=':material/manufacturing:'
                                       )

            # all box size thresholds
            st.divider()
            box_sizes_table(session_state=st.session_state)

        # to slice and dice the data using controls
        relevant_stats = annotation_stats.copy()

        # leave only the data related to selected class
        selected_class = st.session_state.annotations_class_selectbox
        if selected_class != 'All':
            class_indices = (annotation_stats['class_name'] == selected_class)
            relevant_stats = annotation_stats[class_indices]

        # fit the box size range
        box_size_low = st.session_state.box_size_slider[0] / 100  # from percentage to fraction
        box_size_high = st.session_state.box_size_slider[1] / 100

        lower_than_high = (box_size_high >= relevant_stats['box_size'])
        higher_than_low = (box_size_low <= relevant_stats['box_size'])
        relevant_stats = relevant_stats[lower_than_high & higher_than_low]

        # categorize into small, medium and big
        medium_low = st.session_state.medium_low / 100  # from percentage to fraction
        medium_high = st.session_state.medium_high / 100

        bins = [0, medium_low, medium_high, 1]
        labels = ['small', 'medium', 'big']

        relevant_stats['size'] = pd.cut(
            x=relevant_stats['box_size'],
            bins=bins,
            labels=labels,
            right=False
        )

        n_objects = relevant_stats.shape[0]
        if n_objects > configs.ANNOTATIONS_LIMIT:  # no point if less than
            vis_annotations = VisAnnotations(annotation_stats=relevant_stats,
                                             plot_type=st.session_state.plot_type_selectbox,
                                             colormap=colormap)
            vis_annotations.plot_scatters()

            blank_line()

            col1, col2 = st.columns([0.6, 0.4])
            # box sizes histogram
            with col1:
                vis_annotations.plot_distribution()

            # small. medium, big
            with col2:
                vis_annotations.plot_counts()

        else:
            # not enough data animation
            render_html(
                html_filepath='animations/astronaut.html',
                uri=configs.URI['astronaut']
            )

    if stats_page == 'Images':
        widgets_disabled = st.session_state.images_input_page | st.session_state.processing

        with st.sidebar:
            image_classes = copy(classes)
            image_classes.append('background')

            col1, col2 = st.columns(2)
            # image class selector
            col1.selectbox(label='Class',
                           options=image_classes,
                           key='images_class_selectbox',
                           on_change=_backup,
                           kwargs=dict(x='images_class_selectbox'),
                           disabled=widgets_disabled,
                           index=0)

            # class assigning strategy selector
            col2.selectbox(label='Strategy',
                           options=['All', 'Most frequent'],
                           key='images_include_selectbox',
                           on_change=_backup,
                           kwargs=dict(x='images_include_selectbox'),
                           help=configs.HELP_STRATEGY_SELECTBOX,
                           disabled=widgets_disabled,
                           index=0)

        # returns None if key is not found
        image_stats = get_cached_stats(path='.cache/images_stats.h5',
                                       key=st.session_state.data_path)

        # caching options page
        if st.session_state.images_input_page:
            total = images.n_verified
            if st.session_state.images_use_cached_backup:
                total = image_stats.shape[0]

            # update state params based on user choices
            st.session_state = caching_widgets(
                page_name='images',
                cached_exists=image_stats is not None,
                total=total,
                limit=configs.IMAGES_LIMIT,
                session_state=st.session_state)

        # main page
        elif not st.session_state.images_input_page:
            if not st.session_state.images_use_cached_backup:
               image_stats, st.session_state = get_stats(page_name='images',
                                                         getter_function=images.get_stats,
                                                         spinner_message="Scanning images",
                                                         session_state=st.session_state)

            selected_class = st.session_state.images_class_selectbox
            if selected_class != 'All':
                image_assigning_strategy = st.session_state.images_include_selectbox
                class_objects = image_stats['overall']['class_objects']

                if image_assigning_strategy == 'All':
                    # all occurrences
                    relevant_indices = class_objects.apply(
                        lambda x: selected_class in x)
                else:
                    # only if class is the most common class
                    relevant_indices = class_objects.apply(
                        lambda x: selected_class == most_common(x))

                image_stats = image_stats[relevant_indices]

            # at least 10 images
            n_images = image_stats.shape[0]
            if n_images < configs.IMAGES_LIMIT:
                render_html(
                    html_filepath='animations/astronaut.html',
                    uri=configs.URI['astronaut']
                )

            else:
                # tone stats to create color plants
                tone_counts = images.get_tone_counts(image_stats)

                blank_line()
                col1, col2 = st.columns([0.4, 0.6])
                with col1:
                    col1.metric(
                        label='IMAGES',
                        value=n_images
                    )
                    # color planets
                    color_distribution = VisColorDistribution()
                    color_distribution.plot(tone_counts=tone_counts,
                                            x=21, y=57, max_size=10)

                with col2:
                    # images resolutions
                    vis_resolutions = VisResolutions(image_stats=image_stats)
                    vis_resolutions.plot()

                with st.sidebar:
                    # contrast, brightness, saturation
                    cbs_candidates = get_cbs_candidates(image_stats=image_stats, n=3)

                    # returns the selected image path
                    canvas_path = image_select(label='Choose a canvas',
                                               images=cbs_candidates,
                                               use_container_width=True,
                                               captions=None)

                    # generate new candidates
                    _, col = st.columns([0.4, 0.6])
                    col.button(
                        label=':material/refresh:',
                        on_click=get_cbs_candidates.clear,
                        key='refresh_candidates')

                blank_line(3)
                vis_cbs = VisCBS(image_stats=image_stats,
                                 canvas_path=canvas_path)

                vis_cbs.plot()



    if stats_page == 'Overlaps':
        widgets_disabled = st.session_state.overlaps_input_page | st.session_state.processing
        with st.sidebar:
            # Overlap controls
            with st.form('Overlap controls'):
                col1, col2 = st.columns(2)
                # object_1 class selector
                col1.selectbox(label='Relate',
                               options=classes,
                               key='overlaps_object1_selectbox',
                               disabled=widgets_disabled)

                # object_2 class selector
                col2.selectbox(label='With',
                               options=classes,
                               key='overlaps_object2_selectbox',
                               disabled=widgets_disabled)

                st.divider()

                col1, col2 = st.columns(2)
                # number of overlap cases
                col1.slider(label='Cases',
                            min_value=2,
                            max_value=5,
                            key='overlaps_n_cases',
                            disabled=widgets_disabled)

                # number of image examples per case
                col2.slider(label='Images',
                            min_value=1,
                            max_value=5,
                            key='overlaps_n_images',
                            disabled=widgets_disabled)

                blank_line()

                _, col = st.columns([0.33, 0.67])

                col.form_submit_button(label='Pair',
                                       on_click=_backup,
                                       kwargs=dict(
                                           x=['overlaps_object1_selectbox',
                                              'overlaps_object2_selectbox',
                                              'overlaps_n_cases',
                                              'overlaps_n_images']),
                                       icon=':material/compare_arrows:',
                                       disabled=widgets_disabled
                                       )

        overlap_stats = get_cached_stats(path='.cache/overlaps_stats.h5',
                                         key=st.session_state.data_path)

        if st.session_state.overlaps_input_page:
            total = overlaps.n_verified
            if st.session_state.overlaps_use_cached_backup:
                total = overlap_stats['size'].unique()[0]

            st.session_state = caching_widgets(
                page_name='overlaps',
                cached_exists=overlap_stats is not None,
                total=total,
                limit=configs.OVERLAPS_LIMIT,
                session_state=st.session_state)


        # main page
        elif not st.session_state.overlaps_input_page:
            # if there was no cached version
            if not st.session_state.overlaps_use_cached_backup:
                overlap_stats, st.session_state = get_stats(page_name='overlaps',
                                                          getter_function=overlaps.get_stats,
                                                          spinner_message="Calculating overlaps",
                                                          session_state=st.session_state)

            relevant_overlap_stats = overlap_stats.copy()

            # class1
            relate = st.session_state.overlaps_object1_selectbox
            # class2
            relate_with = st.session_state.overlaps_object2_selectbox

            # slicing by class names
            if relate != 'All':
                class_indices = (relevant_overlap_stats['relate'] == relate)
                relevant_overlap_stats = relevant_overlap_stats[class_indices]

            if relate_with != 'All':
                class_indices = (relevant_overlap_stats['relate_with'] == relate_with)
                relevant_overlap_stats = relevant_overlap_stats[class_indices]

            n_overlaps = relevant_overlap_stats.shape[0]
            if n_overlaps < configs.OVERLAPS_LIMIT:
                render_html(
                    html_filepath='animations/astronaut.html',
                    uri=configs.URI['astronaut'])

            else:
                # overlaps and non-overlaps
                non_overlap_indices = (relevant_overlap_stats['overlap_size'] == 0)
                overlap_indices = (relevant_overlap_stats['overlap_size'] > 0)

                non_overlaps = relevant_overlap_stats[non_overlap_indices]
                overlaps = relevant_overlap_stats[overlap_indices]

                # n objects in each of 2 classes
                n_relate_objects = relevant_overlap_stats['relate_index'].nunique()
                n_relate_with_objects = relevant_overlap_stats['with_index'].nunique()

                # single class case
                if relate == relate_with:
                    n_relate_with_objects = 0

                # images with and without overlaps
                n_images_with_overlaps = overlaps['filename'].nunique()
                n_images_without_overlaps = relevant_overlap_stats['filename'].nunique() - n_images_with_overlaps

                vis_overlap = VisOverlap(
                    overlap_stats=relevant_overlap_stats,
                    matching_dict=matching_dict,
                    circle1_name=relate,
                    circle2_name=relate_with,
                    colormap=colormap,
                    n_cases=st.session_state.overlaps_n_cases_backup,
                    n_images=st.session_state.overlaps_n_images_backup)


                vis_overlap.plot()
                blank_line(2)

                col1, col2 = st.columns(2)
                with col1:
                    blank_line(2)
                    col11,col12 = st.columns(2)
                    with col11:
                        plot_overlap_pie(
                            labels=['overlapping', 'not overlapping'],
                            values=[overlaps.shape[0], non_overlaps.shape[0]],
                            height=configs.vh(27),
                            colors=['#F6BD16', '#2F3543'],
                            display_counts=False,
                            title='CO-OCCURRENCES')
                    with col12:
                        plot_overlap_pie(
                            labels=['with overlaps', 'without overlaps'],
                            values=[n_images_with_overlaps, n_images_without_overlaps],
                            height=configs.vh(27),
                            colors=['#E04636', '#2F3543'],
                            display_counts=False,
                            title='IMAGES')
                with col2:
                    vis_overlap_ratios = VisOverlapRatios(colormap=colormap)
                    vis_overlap_ratios.plot(relevant_overlap_stats)

                with st.sidebar:
                    plot_overlap_pie(
                        labels=[f'{relate.capitalize()} objects',
                                f'{relate_with.capitalize()} objects'],
                        values=[n_relate_objects, n_relate_with_objects],
                        height=configs.vh(24),
                        colors=[colormap[relate], colormap[relate_with]],
                        display_counts=not n_relate_with_objects,
                        title='OBJECTS')