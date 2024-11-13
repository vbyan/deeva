import random
from copy import deepcopy
from math import ceil
from bisect import bisect_left
from functools import partial

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans

import configs
from .utils import *
from configs import vw, vh, HUE_RANGES_LOOKUP, HUE_COLORS_REV
from configs.utils import get_image_uri
from utils import render_html


class PlotConfig:
    """
    Remove unnecessary tools from plotly menu bar
    """
    FULLSCREEN = {
        'displaylogo': False,
        'displayModeBar': True,
        'scrollZoom': False,
        'modeBarButtonsToRemove': [
            'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
            'autoScale2d', 'resetScale2d', 'hoverClosestCartesian',
            'hoverCompareCartesian', 'toggleSpikelines'
        ]
    }

    DEFAULT = {
        'displaylogo': False,
        'displayModeBar': False,
        'scrollZoom': False
    }

class Color:
    Primary = "#f5ebeb"
    Secondary = "#a29d9d"
    Stats = '#eddddd'
    Background = '#101010'
    Menu = '#d9aa1f'
    Transparent = 'rgba(0,0,0,0)'

Title = PlotFont('Open Sans', vw(1), "normal", "bold", Color.Primary)
Text = PlotFont('Roboto', vw(0.85), "italic", "normal", Color.Primary)
Legend = PlotFont('Courier New', vw(0.75), "normal", "normal", Color.Stats)
Hover = PlotFont('Courier New', vw(0.9), "normal", "normal", Color.Stats)
AxisTitle = PlotFont('Roboto', vw(0.85), "normal", "normal", Color.Stats)
AxisTick = PlotFont('Roboto', vw(0.9), "italic", "normal", Color.Secondary)
Menu = PlotFont('Roboto', vw(0.75), "normal", "bold", Color.Menu)


class VisMatching:
    """
    Visualizes matching relationships between images and labels.
    Creates an interactive scatter plot showing matched and unmatched items.

     Args:
        matching (dict): Output of DataMatch.get_matching()
        seed (int): Random seed for reproducibility
    """

    def __init__(self, matching: dict, seed: int = 12345):
        self.matching = matching
        self.colormap = {
            'images': '#83c5be',
            'labels': '#ffcb69'
        }
        np.random.seed(seed)

    @staticmethod
    def _generate_points(size: int, x_mean: float, x_std: float,
                         x_bounds: tuple, y_func: callable) -> tuple:
        """
        Generate scattered points with specific distribution characteristics.

        Args:
            size (int): Number of points to generate
            x_mean (float): Mean of x-coordinates
            x_std (float): Standard deviation of x-coordinates
            x_bounds (tuple): (min, max) bounds for x-coordinates
            y_func (callable): Function to generate y-coordinates based on x

        Returns:
            tuple: Arrays of x and y coordinates
        """
        # Generate x coordinates with normal distribution
        x_coords = np.random.normal(loc=x_mean, scale=x_std, size=size)

        # Replace out-of-bounds values with uniform random values
        mask = (x_coords < 0) | (x_coords > 100)
        if np.any(mask):
            x_coords[mask] = np.random.uniform(*x_bounds, size=np.sum(mask))

        # Generate y coordinates using provided function
        y_coords = y_func(x_coords) if size > 0 else []

        return x_coords, y_coords

    @staticmethod
    def _create_hover_text(row: pd.DataFrame) -> str:
        """Create hover text for scatter plot points."""
        return f"{row['color']}: {row['id']}/{row['count']}"

    def _lonely_files_fig(self) -> go.Figure:
        # Generate scattered points for unmatched images (left side)
        x_images, y_images = self._generate_points(
            size=len(self.matching['Lonely images']),
            x_mean=10,
            x_std=30,
            x_bounds=(1, 50),
            y_func=lambda x: np.random.uniform(x + 20, 100)
        )

        # Generate scattered points for unmatched labels (right side)
        x_labels, y_labels = self._generate_points(
            size=len(self.matching['Lonely labels']),
            x_mean=90,
            x_std=30,
            x_bounds=(50, 100),
            y_func=lambda x: np.random.uniform(0, x - 20)
        )

        # Create DataFrame for plotting
        lonely_files_df = pd.DataFrame({
            'x': np.hstack((x_images, x_labels)),
            'y': np.hstack((y_images, y_labels)),
            'size': np.repeat(1, x_images.size + x_labels.size),
            'color': np.hstack((
                np.repeat('Lonely image', x_images.size),
                np.repeat('Lonely label', x_labels.size)
            )),
            'id': np.hstack((
                np.arange(1, x_images.size + 1),
                np.arange(1, x_labels.size + 1)
            )),
            'count': np.hstack((
                np.repeat(x_images.size, x_images.size),
                np.repeat(x_labels.size, x_labels.size)
            ))
        })

        lonely_files_df['hover'] = lonely_files_df.apply(self._create_hover_text, axis=1)

        # Create scatter plot
        fig = px.scatter(
            lonely_files_df,
            x="x", y="y",
            size="size",
            color='color',
            color_discrete_map={'Lonely image' : self.colormap['images'],
                                'Lonely label' : self.colormap['labels']},
            opacity=0.3, hover_name='hover',
            hover_data={k: False for k in lonely_files_df.columns},
        )

        return fig

    def _matched_scatter(self, x: int, y: int, category: str) -> dict:
        return dict(
                x=[x], y=[y],
                opacity=0.8,
                mode='markers+text',
                text=f"{len(self.matching[f'Matched {category}'])}",
                textposition="middle center",
                textfont=dict(size=vw(1.2), color='black', family=Text.family),
                name=f'Matched {category}',
                hovertemplate=f'Matched {category}<extra></extra>',
                marker=dict(
                    size=vw(8),
                    color=self.colormap[category],
                    line=dict(
                        width=vw(0.16),
                        color='DarkSlateGrey')
                ),
                showlegend=False
            )

    def plot(self):
        """
        Create and display the complete matching visualization
        Combines lonely and matched file visualizations.
        """
        # Create base plot with lonely files
        fig = self._lonely_files_fig()

        # Add matched files indicators
        fig.add_scatter(**self._matched_scatter(10, 90, 'images'))
        fig.add_scatter(**self._matched_scatter(90, 10, 'labels'))

        # Configure layout
        fig.update_layout(
            plot_bgcolor=Color.Transparent,
            paper_bgcolor=Color.Transparent,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            hoverlabel=dict(font=Hover.font),
            legend=dict(
                orientation='h',
                xanchor="center",
                x=0.5,
                y=1.1,
                font=Legend.font,
                title_text=''
            ),
            height=vh(45.5)
        )

        # Display plot using Streamlit
        st.plotly_chart(fig, use_container_width=True, config=PlotConfig.DEFAULT)


def plot_general_overview(annotations_counts: list, backgrounds_counts: list,
                          images_counts: list) -> None:
    """
    Plots general overview of annotation formats, image formats and data insights

    Args:
        annotations_counts: list with counts for each annotation format
        backgrounds_counts: list with image and background counts
        images_counts: list with counts for each image format
    """

    annotations_labels = ['yolo', 'voc', 'corrupted labels']
    backgrounds_labels = ['images', 'backgrounds']
    images_labels = ['png', 'jpg', 'jpeg', 'corrupted images']

    annotations_colors = {label: color for label, color in
                          zip(annotations_labels, ['#F4A300', '#D83F6C', '#6A1B9A'])}
    backgrounds_colors = {label: color for label, color in
                          zip(backgrounds_labels, ['#736565', '#46b3bd'])}
    images_colors = {label: color for label, color in
                     zip(images_labels, ['#8BC34A', '#FF5722', '#3F51B5', '#9C27B0'])}

    titles = ['Labels', 'Data', 'Images']

    # all 3 pie charts
    pies = [go.Pie(
        labels=[label for label, value in zip(labels, counts) if value > 0],
        values=[value for value in counts if value > 0],
        marker=dict(colors=[colors[label] for label, value in zip(labels, counts) if value > 0],
                    line=dict(color=Color.Background, width=vw(0.2))),
        hoverinfo='label + value',
        textinfo='text',
        text=[f'{(value / sum(counts)) * 100:.3g}%' for value in counts if value > 0],
        textposition='outside',
        hole=.88,
        sort=False,
        opacity=1,
        title=title,
        titlefont=dict(size=vw(1.1), color=Color.Secondary, family=Text.family,
                       style=Text.style, weight=Title.weight),
        outsidetextfont=Text.font
    ) for labels, counts, colors, title in
        zip([annotations_labels, backgrounds_labels, images_labels],
            [annotations_counts, backgrounds_counts, images_counts],
            [annotations_colors, backgrounds_colors, images_colors],
            titles)]

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[None, {'type': 'domain'}],
               [{'type': 'domain'}, None],
               [None, {'type': 'domain'}]]
    )

    fig.add_trace(pies[0], row=1, col=2)
    fig.update_traces(domain=dict(x=[0.5, 0.9], y=[0.6, 1]), row=1, col=2)

    fig.add_trace(pies[1], row=2, col=1)
    fig.update_traces(domain=dict(x=[0.1, 0.5], y=[0.3, 0.7]), row=2, col=1)

    fig.add_trace(pies[2], row=3, col=2)
    fig.update_traces(domain=dict(x=[0.5, 0.9], y=[0, 0.4]), row=3, col=2)

    fig.update_layout(
        title='General  Overview',
        title_font=Title.font,
        title_x=0,
        title_y=0.95,
        plot_bgcolor=Color.Transparent,
        paper_bgcolor=Color.Transparent,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.45,
            xanchor="right",
            x=1.2,
            font=Legend.font
        ),
        hoverlabel=dict(font=Hover.font),
        autosize=False,
        height=vh(80),
        margin=go.layout.Margin(
            l=0,
            r=0,
            b=vh(5),
            t=0,
            pad=0
        ),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


class VisClasses:
    """
    Visualizes class distributions. Supports both instance counts and image counts per class,
    with customizable thresholds for highlighting scarce classes.

    Args:
        annotation_stats: Output of Annotations.get_stats() method
        max_classes: Maximum number of classes to display
        threshold: Threshold ratio for highlighting scarce classes
        threshold_type: Type of threshold calculation ('max' or 'total')
    """

    def __init__(self, annotation_stats: pd.DataFrame, max_classes: int = 10,
                 threshold: float = 0.1, threshold_type: str = 'max'):

        self.annotation_stats = annotation_stats
        self.max_classes = max_classes
        self.threshold = threshold
        self.threshold_type = threshold_type

        # Visual styling constants
        self.colors = {
            'Instances': '#3ec024',
            'Images': '#B113D1',
            'Scarce': '#D62728'
        }

        # Prepare the data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare and transform the data for plotting."""
        # Get class counts and top classes
        class_counts = self.annotation_stats['class_name'].value_counts()
        top_classes = class_counts.head(self.max_classes)

        # Create merged DataFrame
        self.df_merged = pd.DataFrame({
            'Class': top_classes.index,
            'Instances': top_classes.values,
            'Images': self.annotation_stats.groupby('class_name')['filename']
            .nunique()[top_classes.index]
        })

        # Prepare plot data
        self.df_plot = pd.melt(
            self.df_merged,
            id_vars=['Class'],
            value_vars=['Instances', 'Images'],
            var_name='Metric',
            value_name='Count'
        )

        # Calculate thresholds
        counts_by_metric = self.df_plot.groupby('Metric')['Count']
        self.thresholds = (
                              counts_by_metric.max() if self.threshold_type == 'max'
                              else counts_by_metric.sum()
                          ) * self.threshold

    def _create_bar_trace(self, data: pd.DataFrame, metric: str,
                          is_scarce: bool, visible: bool) -> go.Bar:
        return go.Bar(
            x=data['Count'], y=data['Class'],
            name="Scarce" if is_scarce else "Normal",
            orientation='h',
            text=data['Count'],
            textposition='inside',
            textangle=0,  # Force horizontal text
            texttemplate='%{text:.2s}',
            insidetextanchor='middle',
            insidetextfont=dict(size=vw(0.75), weight='bold'),
            marker=dict(
                color=self.colors['Scarce' if is_scarce else metric],
                pattern=dict(shape="/" if is_scarce else "", size=vw(0.25), solidity=0.5)
            ),
            hovertemplate=(
                    '<b>%{y}</b> <b>' + metric.lower() + '</b>: %{x}<br>'
                    '<b>ratio total:</b> %{customdata[0]:.2g}<br>'
                    '<b>ratio max:</b> %{customdata[1]:.2g}'
                    '<extra></extra>'
            ),
            customdata=data[['Ratio', 'Ratio_max']],
            hoverlabel=dict(align='left', font=Hover.font),
            visible=visible)

    def _get_xaxis(self, metric: str) -> dict:
        return dict(title=metric,
                    range=[0, self.df_plot[self.df_plot['Metric'] == metric]['Count'].max() * 1.1],
                    title_text='Count',
                    title_font=AxisTitle.font,
                    tickfont=AxisTick.font,
                    showticklabels=False)


    def plot(self) -> list:
        """
        Create and display the visualization.
        """
        fig = go.Figure()

        # Create traces for each metric
        for metric in ['Instances', 'Images']:
            metric_data = self.df_plot[self.df_plot['Metric'] == metric].copy()
            metric_data['Ratio_max'] = metric_data['Count'] / metric_data['Count'].max()
            metric_data['Ratio'] = metric_data['Count'] / metric_data['Count'].sum()

            # Split data into normal and scarce based on threshold
            scarce_mask = metric_data['Count'] < self.thresholds[metric]

            for is_scarce in [False, True]:
                data = metric_data[scarce_mask if is_scarce else ~scarce_mask]
                if not data.empty:
                    fig.add_trace(self._create_bar_trace(
                        data=data,
                        metric=metric,
                        is_scarce=is_scarce,
                        visible=(metric == 'Instances')
                    ))
                else:
                    # Add empty placeholder trace
                    fig.add_trace(go.Bar(
                        x=[], y=[],
                        name="Placeholder",
                        visible=(metric == 'Instances')
                    ))

        fig.update_layout(
            plot_bgcolor=Color.Transparent,
            paper_bgcolor=Color.Transparent,
            height=vh(40),
            bargap=0.3, bargroupgap=0.1,
            barcornerradius=15,

            legend=dict(
                x=-0.4, y=0,
                font=Legend.font,
                title=None
            ),

            hoverlabel=dict(namelength=-1, font=Hover.font),

            xaxis1=self._get_xaxis('Instances'),
            xaxis2=self._get_xaxis('Images'),
            yaxis=dict(
                title_text='Class',
                title_font=AxisTitle.font,
                tickfont=AxisTick.font,
                showgrid=False,
                type='category'
            ),

            updatemenus=[dict(
                buttons=[
                    dict(label="Instances", method="update",
                         args=[{"visible": [True, True, False, False], "xaxis": "x1"}]),
                    dict(label="Images", method="update",
                         args=[{"visible": [False, False, True, True] , "xaxis": "x2"}])
                ],
                direction="down",
                pad={"r": vw(1), "t": 0},
                showactive=False,
                x=-0.4, xanchor="left",
                y=1, yanchor="top",
                bgcolor=Color.Transparent,
                font=Menu.font,
                bordercolor='#404040',
        )],

            margin=dict(l=vw(4.7), r=vw(1), b=vh(9.3), t=vh(0.93))
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True, config=PlotConfig.FULLSCREEN)

        relevant_classes = self.df_merged['Class'].tolist()
        return relevant_classes


class VisCoOccurrences:
    """
    Visualizes co-occurrences between classes. Supports both total and normalized counts

    Args:
        annotation_stats: Output of Annotations.get_stats() method
        relevant_classes: list of classes to include
    """
    def __init__(self, annotation_stats: pd.DataFrame, relevant_classes: list):
        self.annotation_stats = annotation_stats
        self.relevant_classes = relevant_classes

    def _get_co_occurrences(self):
        """Get total counts"""
        relevant_indices = self.annotation_stats['class_name'].isin(self.relevant_classes)
        annotation_stats = self.annotation_stats[relevant_indices]

        occurrences = pd.crosstab(annotation_stats['filename'], annotation_stats['class_name'])
        binary = occurrences.map(lambda x: 1 if x > 0 else 0) # occurs with or not

        self_occurrences = np.sum((occurrences > 1), axis=0) # class occurring with itself
        co_occurrences = binary.T.dot(binary) # class occurring with other classes

        np.fill_diagonal(co_occurrences.values, self_occurrences) # fill diagonal values with self-occurrences

        return co_occurrences


    def _get_normalized(self, co_occurrences: pd.DataFrame) -> pd.DataFrame:
        """Get normalized counts"""
        image_counts = self.annotation_stats.groupby('class_name')['filename'].nunique()
        image_counts_sorted = image_counts.loc[co_occurrences.index]

        # maximum possible co-occurrences between two classes
        max_possible = np.outer(image_counts_sorted, image_counts_sorted)

        # normalizing by class_frequency
        normalized = co_occurrences / max_possible
        # normalizing as a fraction of total co-occurrences
        normalized = normalized / normalized.values.sum()

        return normalized

    @staticmethod
    def _get_heatmap_trace(df: pd.DataFrame):
        return go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale=[Color.Transparent] + px.colors.sequential.Electric,
            visible=True,  # Show this trace by default
            hovertemplate="%{z}<br><br>x: %{x}<br>y: %{y}<extra></extra>",
            hoverlabel=dict(align='left', font=Hover.font),
            colorbar=dict(len=0.7, x=1.1, tickvals=[], ticktext=[]))

    def plot(self):
        co_occurrences = self._get_co_occurrences()
        normalized = self._get_normalized(co_occurrences)

        co_occurrences = co_occurrences.iloc[::-1, :]
        normalized = normalized.iloc[::-1, :]

        fig = go.Figure()

        # total trace
        fig.add_trace(self._get_heatmap_trace(co_occurrences))
        # normalized trace
        normalized_trace = self._get_heatmap_trace(normalized)
        normalized_trace.update(visible=False)
        fig.add_trace(normalized_trace)

        # Define the layout and dropdown menu
        fig.update_layout(
            coloraxis_colorbar=dict(title="Co-occurrence Frequency",
                                    nticks=3),
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label="Total",
                            method="update",
                            args=[{"visible": [True, False]}]  # Show Total and hide Normalized
                        ),
                        dict(
                            label="Normalized",
                            method="update",
                            args=[{"visible": [False, True]}]  # Hide Total and show Normalized
                        )
                    ],
                    direction="left",
                    pad={"r": vw(1), "t": 0},
                    showactive=False,
                    x=1.2,
                    xanchor="right",
                    y=1.2,
                    yanchor="top",
                    bgcolor=Color.Transparent,
                    font=Menu.font,
                    bordercolor='#404040',
                )
            ],
            plot_bgcolor=Color.Transparent,
            paper_bgcolor=Color.Transparent,
            font=Text.font,
            legend=dict(x=-0.4, y=0, font=Legend.font),
            margin=dict(l=0,r=0,b=0,t=vh(6)),
            hoverlabel=dict(namelength=-1,font=Hover.font),
            height=vh(41)
        )

        fig.update_xaxes(title_text='', title_font=AxisTitle.font, tickfont=AxisTick.font)
        fig.update_yaxes(title_text='', title_font=AxisTitle.font, tickfont=AxisTick.font, showgrid=False)

        fig.update_layout(title='Co-occurrence Matrix', title_font=Title.font)
        st.plotly_chart(fig, use_container_width=True, config=PlotConfig.DEFAULT)


class VisAnnotations:
    """
    A class for visualizing annotation statistics using various plot types.

    This class provides methods to create the following charts
        - bbox center coordinates scatter
        - bbox width/height relationship scatter
        - median bbox filled scatter
        - bbox size distribution histogram
        - bbox size categories ('big', 'medium', 'small') pie

    Methods:
        plot_scatters: visualizes center, width/height and medium bbox scatters
        plot_distribution: plots a histogram for bbox sizes
        plot_counts: plots a pie chart for bbox size categories
    """

    def __init__(self, annotation_stats: pd.DataFrame, plot_type: str, colormap: dict):
        self.annotation_stats = annotation_stats
        self.plot_type = plot_type
        self.colormap = colormap
        self.n_data = annotation_stats.shape[0]
        self.opacity, self.size_max, self.marker_line_width = self._calculate_scatter_params(self.n_data)

    def plot_scatters(self) -> None:
        """
        Plot the following scatter plots
            - bbox center coordinates scatter
            - bbox width/height relationship scatter
            - median bbox filled scatter
        """

        fig = make_subplots(rows=1, cols=3, shared_xaxes=False, shared_yaxes=False,
                            vertical_spacing=0.1)

        fig.update_layout(**self._create_base_layout(width=vw(21), height=vh(32), show_legend=True))

        # Add the three different visualizations
        self._add_plot_data(fig, 'x_center', 'y_center', row=1, col=1, show_legend=not self._is_single_class())
        self._add_plot_data(fig, 'width', 'height', row=1, col=2, show_legend=False)
        self._add_box_visualization(fig, row=1, col=3)

        # Update layout

        # Set axis titles
        fig.update_xaxes(title='X-center', title_font=AxisTitle.font, row=1, col=1)
        fig.update_yaxes(title='Y-center', title_font=AxisTitle.font, row=1, col=1)
        fig.update_xaxes(title='Width', title_font=AxisTitle.font, row=1, col=2)
        fig.update_yaxes(title='Height', title_font=AxisTitle.font, row=1, col=2)

        # Update axes properties
        self._update_axes(fig)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True, config=PlotConfig.DEFAULT)

    def plot_distribution(self) -> None:
        """Plot bbox sizes histogram"""
        if not self._is_single_class():
            fig = px.histogram(self.annotation_stats, x="box_size", nbins=20, color="class_name",
                               color_discrete_map=self.colormap, hover_data={'box_size': False})

            fig.update_layout(**self._create_base_layout(vw(36.5), height=vh(43)))

            # Add dropdown menu to the histogram using Plotly update buttons
            fig.update_layout(
                updatemenus=[
                    dict(
                        buttons=list([
                            dict(label="Counts",
                                 method="update",
                                 args=[{"barnorm": 'counts'}, {"barnorm": 'counts'}]),
                            dict(label="Fraction",
                                 method="update",
                                 args=[{"barnorm": 'fraction'}, {"barnorm": 'fraction'}])
                        ]),
                        direction="down",
                        pad={"r": vw(1), "t": 0},
                        showactive=False,
                        x=-0.3,
                        xanchor="left",
                        y=1,
                        yanchor="top",
                        bgcolor=Color.Transparent,
                        font=Menu.font,
                        bordercolor='#404040',
                    ), ],
                yaxis_title='Fraction',
                legend=dict(
                    title='Class',
                    orientation="v",
                    y=1,
                    x=1.1,
                    font=Legend.font,
                ),
                showlegend=True
            )

        else:
            fig = px.histogram(self.annotation_stats, x="box_size", nbins=20, color='class_name',
                               color_discrete_map=self.colormap, marginal='violin',
                               hover_data={'class_name':False, 'box_size': False})

            fig.update_layout(
                yaxis_title='Count',
                **self._create_base_layout(vw(36.5), height=vh(43))
            )

        fig.update_xaxes(title='Box size', showgrid=False, title_standoff=vw(2),
                         title_font=AxisTitle.font)
        fig.update_yaxes(showgrid=False, title_standoff=vw(3), title_font = AxisTitle.font)

        fig.update_traces(marker_line_width=vw(0.05), marker_line_color="black")

        fig.update_layout(
            xaxis2=dict(showticklabels=False, title=None),
            yaxis2=dict(showticklabels=False, title=None),
            margin=dict(l=0, r=0, b=0, t=vh(4)),
            legend = dict(title=None)
        )

        st.plotly_chart(fig, use_container_width=True, config=PlotConfig.DEFAULT)

    def plot_counts(self) -> None:
        """Plot bbox sizes pie chart"""
        data = self.annotation_stats['size'].value_counts()
        values = data.values
        labels = data.index

        total = sum(data.values)

        colors = {
            'small': '#FFB100',  # Bright amber/orange
            'medium': '#00E5FF',  # Vivid cyan
            'big': '#1FE36C'  # Vibrant green
        }
        relevant_colors = [colors[key] for key in labels]

        text = [value if value > 0 else '' for value in values]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=relevant_colors, line=dict(color='#131313', width=vw(0.5))),
                    textfont=dict(size=vw(1.1)),
                    hoverinfo='label+percent',
                    textinfo='text',
                    text=text,
                    hole=.4
                )
            ]
        )
        fig.update_layout(**self._create_base_layout(width=vw(35), height=vh(35)))

        fig.update_layout(
            legend=dict(
                orientation="v",  # Place the legend horizontally below the chart
                yanchor="bottom",
                y=0.8,
                xanchor="right",
                x=1.2,
                font=Legend.font,
            ),
            annotations=[
                dict(
                    text=str(total),  # The text you want to display
                    showarrow=False,
                    font=dict(size=vw(1.2), color=Color.Secondary),  # Adjust the size as needed
                    x=0.5,
                    y=0.5,
                    xanchor='center',
                    yanchor='middle'
                )],
            showlegend=True,
            margin=dict(
                l=vw(4),
                t=vh(8),
                r=0, b=0, pad=0
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _calculate_scatter_params(n_points: int) -> tuple:
        size_max = max(2, 18 - np.log2(n_points))
        marker_line_width = size_max / 5

        if n_points < 1000:
            opacity = 0.55
        else:
            # Logarithmic scaling for larger datasets
            opacity = 1 / (np.emath.logn(1000, n_points) + 1)
            opacity = max(0.25, opacity)

        return opacity, size_max, marker_line_width

    def _is_single_class(self) -> bool:
        return self.annotation_stats['class_name'].nunique() == 1

    @staticmethod
    def _create_base_layout(width: float, height: float, show_legend: bool=False) -> dict:
        return dict(
            autosize=False,
            plot_bgcolor=Color.Transparent,
            paper_bgcolor=Color.Transparent,
            width=width,
            height=height,
            margin= dict(
                l=0, r=0, b=0,
                t=vh(2),
                pad=vh(2)
            ),
            font=Text.font,
            showlegend=show_legend,
            hoverlabel=dict(font=Hover.font),
            legend=dict(
                itemsizing='constant',  # Makes legend items constant size
                itemwidth=30,  # Controls legend icon size
                font=Legend.font)
        )

    def _create_scatter_trace(self, x: str, y: str, show_legend: bool = True) -> go.Figure:
        scatter = px.scatter(
            self.annotation_stats,
            x=x, y=y,
            color='class_name',
            opacity=self.opacity,
            color_discrete_map=self.colormap,
            hover_name='class_name',
            size=[1] * self.n_data,  # Set uniform size for all points
            size_max=self.size_max,
            hover_data={x: ':.2f',
                        y: ':.2f',
                        'class_name': False}
        )

        for trace in scatter['data']:
            trace.update(
                legendgroup=trace['name'],
                showlegend=show_legend,
                marker=dict(
                    line=dict(
                        width=self.marker_line_width,
                        color='black'
                    )
                ))

        return scatter

    def _create_heatmap_trace(self, x: str, y: str) -> go.Figure:
        return px.density_heatmap(
            self.annotation_stats,
            x=x, y=y,
            nbinsx=10,
            histnorm='percent',
            histfunc='count',
            color_continuous_scale=[Color.Transparent] + px.colors.sequential.thermal
        )

    def _add_plot_data(self, fig: go.Figure, x: str, y: str, row: int,
                       col: int, show_legend: bool = True) -> None:
        if self.plot_type == 'Scatter':
            scatter = self._create_scatter_trace(x, y, show_legend)
            for trace in scatter['data']:
                fig.add_trace(trace, row=row, col=col)
        else:
            heatmap = self._create_heatmap_trace(x, y)
            for trace in heatmap.data:
                fig.add_trace(trace, row=row, col=col)
            fig.update_layout(
                coloraxis={
                    'colorscale': [Color.Transparent] + px.colors.sequential.thermal,
                    'showscale': True
                }
            )

    def _add_box_visualization(self, fig: go.Figure, row: int, col: int) -> None:
        size_groups  = self.annotation_stats.groupby(['class_name', 'size'], observed=False)
        rectangles = size_groups[['width', 'height']].median().reset_index().dropna(axis=0)

        fill_alpha = dict(big=0.5, medium=0.7, small=0.9)
        fill_factor = dict(big=0.4, medium=0.5, small=0.6)

        for _, rect in rectangles.iterrows():
            color = self.colormap[rect['class_name']]
            line_color = hex_to_rgba(color, alpha=1, factor=1)
            fillcolor = hex_to_rgba(color, alpha=fill_alpha[rect['size']],
                                    factor=fill_factor[rect['size']])

            x_center = random.uniform(0.3, 0.7)
            y_center = random.uniform(0.3, 0.7)
            half_width = rect['width'] / 2
            half_height = rect['height'] / 2

            x0 = x_center - half_width
            x1 = x_center + half_width
            y0 = y_center - half_height
            y1 = y_center + half_height

            fig.add_trace(
                go.Scatter(
                    x=[x0, x0, x1, x1, x0],
                    y=[y0, y1, y1, y0, y0],
                    fill="toself",
                    mode='lines',
                    fillcolor=fillcolor,
                    line=dict(color=line_color, width=2),
                    hoverinfo='text',
                    text=f"{rect['size'].capitalize()} {rect['class_name']}",
                    hoverlabel=dict(font=Hover.font),
                    opacity=0.7,
                    legendgroup=rect['class_name'],
                    showlegend=False
                ),
                row=row,
                col=col
            )

    @staticmethod
    def _update_axes(fig: go.Figure) -> None:
        for i in range(1, 4):
            for axis in ['xaxis', 'yaxis']:
                fig.update_layout({
                    f"{axis}{i}": {
                        'showgrid': False,
                        'showticklabels': False,
                        'title_standoff': vw(1.2),
                        'range': [0, 1]
                    }
                })


class VisColorDistribution:
    """
    Visualizes color distributions as planets

    This class creates planet-like visualizations based on color tones,
    using heatmaps to represent the distribution of saturation and value.

     Args:
        max_tones (int): Maximum number of unique tones to use for plotting
        sampling_factor (int): Factor used for creating tone grid
    """

    def __init__(self, max_tones: int=1000, sampling_factor: int=5):
        self.max_tones = max_tones
        self.sampling_factor = sampling_factor

    @st.cache_data(show_spinner=False)
    def plot(_self, tone_counts: pd.DataFrame, x: int, y: int, max_size: int) -> None:
        """
        Plot the color distribution as planets.

        Args:
            tone_counts: pd.DataFrame containing color tone information.
            x: Starting x-coordinate for planet placement.
            y: Y-coordinate for planet placement.
            max_size: Maximum size of the largest planet.
        """
        # draw a planet for each hue (color) group
        biggest_planet_size = 0

        for hue in tone_counts['hue'].unique():
            # select tones corresponding to selected hue (color)
            color_tone_indices = tone_counts['hue'] == hue
            color_tones = tone_counts[color_tone_indices]

            color = HUE_COLORS_REV[HUE_RANGES_LOOKUP[int(hue * 180)]] # converting from 0-1 to 0-180

            # future planet size
            size = color_tones['count'].sum() / tone_counts['count'].sum()
            # planet texture
            texture = _self._get_texture(color_tones)

            if size > biggest_planet_size:
                biggest_planet_size = size

            # if the planet size is at least one-fourth of the biggest
            if size >= biggest_planet_size / 3:
                # actual size to render
                true_size = max_size / biggest_planet_size * size
                render_html('animations/static_planet.html',
                            uri=get_image_uri(texture),
                            size=true_size,
                            x=x, y=y,
                            rotation_time=30,
                            planet_name=color,
                            color=Color.Secondary,
                            key=f'planet_{color}')

                # move x-coordinate incrementally to the right
                x += true_size / 2 + max_size / 6


    def _get_texture(self, color_tones):
        """Returns the planet texture for a selected color"""
        z, value_sat_ratio = self._get_contour_z(color_tones)

        hue = color_tones['hue'].iloc[0] # same hue in entire color_tones
        colorscale = self._get_colorscale(hue, z, value_sat_ratio)

        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=z, colorscale=colorscale,
                          zsmooth='best', showscale=False))

        fig.update_layout(
            width=1000,
            height=1000,
            plot_bgcolor=Color.Transparent,
            paper_bgcolor=Color.Transparent,
            showlegend=False)

        fig.update_xaxes(showticklabels=False, showgrid=False, showline=False, ticks='')
        fig.update_yaxes(showticklabels=False, showgrid=False, showline=False, ticks='')

        im_arr = plotly_fig2array(fig)
        planet_texture = get_planet_texture(im_arr)

        return planet_texture

    def _get_contour_z(self, color_tones, sat_ratio=0.5):
        # number of tones overall and et each row
        n_samples = self._get_n_samples(color_tones)
        n_z_cols = n_samples // self.sampling_factor

        color_tones = color_tones.sample(n_samples)

        # mixing saturation and value channels
        # rgb image
        if np.any(color_tones['sat']) > 0:
            mix = sat_ratio * color_tones['sat'] + (1 - sat_ratio) * color_tones['value']
            z = mix.sort_values(ascending=True).values.reshape(-1, n_z_cols)

            value_sat_ratio = color_tones.value.mean() / color_tones.sat.mean()

        # grayscale image
        else:
            z = color_tones['value'].sort_values(ascending=True).values.reshape(-1, n_z_cols)

            value_sat_ratio = 0

        return z, value_sat_ratio

    def _get_n_samples(self, color_tones):
        color_samples = color_tones.shape[0]

        n_samples = min(color_samples, self.max_tones)
        n_samples = n_samples // self.sampling_factor * self.sampling_factor

        return n_samples

    @staticmethod
    def _get_colorscale(hue, z, value_sat_ratio):
        z_row_values =  np.linspace(np.min(z), np.max(z), 11) # 11 rows
        # grayscale image
        if not value_sat_ratio:
            color_codes = [hsv_to_hex(hue, 0, x) for x in z_row_values]
        else:
            color_codes = [hsv_to_hex(hue, x, min(1, value_sat_ratio * x)) for x in z_row_values]

        # Create a colorscale as a list of pairs
        colorscale = [(s, c) for s, c in zip(np.linspace(0, 1, 11), color_codes)]

        return colorscale


class VisCBS:
    """
    Visualizes contrast, brightness and saturation distribution on a canvas

    Args:
       image_stats: pd.DataFrame containing image statistics
       canvas_path: Path to the canvas image file
       n_segments: Number of segments to have on the canvas
    """
    def __init__(self, image_stats: pd.DataFrame, canvas_path: str, n_segments: int=5):
        canvas = cv2.imread(canvas_path)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = cv2.resize(canvas, (180, 200))

        self.stacked_canvas = np.tile(canvas, (n_segments, 1))
        self.stacked_canvas_gray = cv2.cvtColor(self.stacked_canvas, cv2.COLOR_BGR2GRAY)
        self.stacked_canvas_hsv = cv2.cvtColor(self.stacked_canvas, cv2.COLOR_RGB2HSV)

        self.canvas_contrast = np.std(self.stacked_canvas_gray)
        self.canvas_brightness = np.mean(self.stacked_canvas_gray)
        self.canvas_saturation = np.mean(cv2.split(self.stacked_canvas_hsv)[1])

        self.h, self.w = self.stacked_canvas.shape[:2]

        self.contrast_values = self._sample(image_stats['overall']['RMS_contrast'].values)
        self.brightness_values = self._sample(image_stats['overall']['brightness'].values)
        self.saturation_values = self._sample(image_stats.xs('sat', level=1, axis=1).mean(axis=1).values)

        self.colormap = dict(saturation='#18E7CB',
                             brightness='#CB18E7',
                             contrast='#E7CB18')

    def plot(self) -> None:
        """Plot all 3 distributions as update-menus options with synchronized vertical lines"""
        fig = make_subplots(
            rows=2, cols=2, row_heights=[0.8, 0.2], column_widths=[0.8, 0.2],
            vertical_spacing=0,
            specs=[[{}, {"rowspan": 2}], [{}, None]],
            shared_xaxes=True,
        )

        # canvases
        contrast_canvas = self._postprocess(self._get_contrast_canvas())
        brightness_canvas = self._postprocess(self._get_brightness_canvas())
        saturation_canvas = self._postprocess(self._get_saturation_canvas())

        # fit into [0-100] range
        contrast_values = [value / 127.5 * 100 for value in self.contrast_values]
        brightness_values = [value / 255 * 100 for value in self.brightness_values]
        saturation_values = [value / 255 * 100 for value in self.saturation_values]

        # Generate the image figures
        img_contrast = px.imshow(contrast_canvas)
        img_brightness = px.imshow(brightness_canvas)
        img_saturation = px.imshow(saturation_canvas)

        # img hovers
        img_contrast.update_traces(hovertemplate="Contrast distribution<extra></extra>")
        img_brightness.update_traces(hovertemplate="Brightness distribution<extra></extra>")
        img_saturation.update_traces(hovertemplate="Saturation distribution<extra></extra>")

        # Add the current chart traces to the first row
        fig.add_trace(img_contrast.data[0], row=1, col=1)
        fig.add_trace(img_brightness.data[0], row=1, col=1)
        fig.add_trace(img_saturation.data[0], row=1, col=1)

        # origins
        fig.add_trace(self._get_origin_trace(self.contrast_values, self.canvas_contrast), col=1, row=1)
        fig.add_trace(self._get_origin_trace(self.brightness_values, self.canvas_brightness), col=1, row=1)
        fig.add_trace(self._get_origin_trace(self.saturation_values, self.canvas_saturation, visible=True), col=1, row=1)

        # histograms
        fig.add_trace(self._get_hist_trace('contrast', contrast_values), row=1, col=2)
        fig.add_trace(self._get_hist_trace('brightness', brightness_values), row=1, col=2)
        fig.add_trace(self._get_hist_trace('saturation', saturation_values, visible=True), row=1, col=2)

        # area charts
        fig.add_trace(self._get_area_trace('contrast', contrast_values), row=2, col=1)
        fig.add_trace(self._get_area_trace('brightness', brightness_values), row=2, col=1)
        fig.add_trace(self._get_area_trace('saturation', saturation_values, visible=True), row=2, col=1)

        fig.update_layout(
            height=vh(32),
            plot_bgcolor=Color.Transparent,
            paper_bgcolor=Color.Transparent,
            margin=dict(r=0, l=0, t=vw(2), b=0),
            hoverlabel=dict(font=Hover.font),
            legend=dict(font=Legend.font, x=-0.1, y=0),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict( label="Saturation",method="update",
                            args=[{"visible": [False, False, True] * 4}]),
                        dict(label="Contrast", method="update",
                             args=[{"visible": [True, False, False] * 4}]),
                        dict(label="Brightness", method="update",
                             args=[{"visible": [False, True, False] * 4}]),
                    ],
                    direction='down',
                    pad={"r": vw(1)},
                    showactive=True,
                    x=-0.1,
                    xanchor="left",
                    y=0.65,
                    yanchor="top",
                    bgcolor='rgb(30,30,30)',
                    font=Menu.font,
                    bordercolor='#404040'
                )
            ]
        )

        fig.update_traces(xaxis="x1", row=1, col=1)
        fig.update_traces(xaxis="x1", row=2, col=1)

        fig.update_xaxes(showticklabels=False, showgrid=False, showline=False, ticks='', row=1, col=1,
                         showspikes=True, spikemode='across', spikesnap='cursor', spikecolor='black',
                         spikethickness=vw(0.08), spikedash='dot')
        fig.update_yaxes(showticklabels=False, showgrid=False, showline=False, ticks='', row=1, col=1)

        fig.update_xaxes(showgrid=False, showline=False, col=2, range=[0, 100],
                         tickfont=AxisTick.font, tickvals=[25, 50, 75, 100])
        fig.update_yaxes(showgrid=False, showline=False, col=2, nticks=5,
                         tickfont=AxisTick.font)

        fig.update_yaxes(side='right', showgrid=False, showline=False, tickfont=AxisTick.font,
                         dtick=30, row=2, col=1)
        fig.update_xaxes(showticklabels=False, showgrid=False, showline=False, row=2, col=1)



        st.plotly_chart(fig, use_container_width=True, config=PlotConfig.DEFAULT)


    def _sample(self, arr: np.ndarray) -> list:
        return sorted(np.random.choice(arr, size=self.w, replace=True))

    def _get_contrast_canvas(self) -> np.ndarray:
        """Generate a canvas representing contrast distribution"""
        std_dev = np.std(self.stacked_canvas)

        contrast_array = np.repeat(self.contrast_values, self.h, axis=0).reshape(self.w, self.h).T
        contrast_array = np.repeat(contrast_array[:, :, np.newaxis], 3, axis=2)

        standard_scaled = (self.stacked_canvas - np.mean(self.stacked_canvas)) / std_dev
        contrast_canvas = standard_scaled * contrast_array + np.mean(self.stacked_canvas)

        return contrast_canvas

    def _get_brightness_canvas(self) -> np.ndarray:
        """Generate a canvas representing brightness distribution"""
        stacked_canvas_lab = cv2.cvtColor(self.stacked_canvas, cv2.COLOR_RGB2Lab)
        l_channel, a_channel, b_channel = cv2.split(stacked_canvas_lab)

        brightness_array = np.repeat(self.brightness_values, self.h, axis=0).reshape(self.w, self.h).T

        l_channel = l_channel - np.mean(self.stacked_canvas_gray) + brightness_array
        l_channel = np.clip(l_channel, 0, 255).astype('uint8')

        brightness_canvas_lab = cv2.merge([l_channel, a_channel, b_channel])
        brightness_canvas = cv2.cvtColor(brightness_canvas_lab, cv2.COLOR_Lab2RGB)

        return brightness_canvas

    def _get_saturation_canvas(self) -> np.ndarray:
        """Generate a canvas representing saturation distribution"""
        h_channel, s_channel, v_channel = cv2.split(self.stacked_canvas_hsv)
        s_mean = s_channel.mean()

        saturation_array = np.repeat(self.saturation_values, self.h, axis=0).reshape(self.w, self.h).T

        s_channel = s_channel / s_mean * saturation_array
        s_channel = np.clip(s_channel, 0, 255).astype('uint8')

        saturation_canvas_hsv = cv2.merge([h_channel, s_channel, v_channel])
        saturation_canvas = cv2.cvtColor(saturation_canvas_hsv, cv2.COLOR_HSV2RGB)

        return saturation_canvas

    @staticmethod
    def _postprocess(canvas: np.ndarray) -> np.ndarray:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        width = ceil(vw(0.1))
        canvas = cv2.copyMakeBorder(
            canvas, width, width, width, width,
            cv2.BORDER_CONSTANT, value=[64, 64, 64])

        return canvas

    def _get_hist_trace(self, label:str, values: list, visible: bool=False):
        return go.Histogram(x=values, nbinsx=30,
                            marker=dict(
                                color=self.colormap[label],
                                line=dict(width=2, color='black'),
                                pattern=dict(shape="-")  # Optional: Adds a pattern fill to the bars
                            ),
                            hovertemplate='Count: %{y}<extra></extra>',
                            showlegend=False,
                            visible=visible)

    def _get_area_trace(self, label: str, values: list, visible: bool=False):
        trace = go.Scatter(y=sorted(values),
                          mode='lines+markers',  # Include markers in the mode
                          fill='tozeroy',
                          line=dict(
                              color=self.colormap[label],
                          ),
                          fillcolor=hex_to_rgba(self.colormap[label], alpha=0.5, factor=0.5),
                          visible=visible,
                          text=[label] * len(values),
                          hovertemplate='%{text}: %{y:.0f}/100<extra></extra>',
                          marker=dict(size=vw(0.15)),
                          showlegend=False
                          )

        return trace

    def _get_origin_trace(self, dataset_values: list, canvas_value, visible: bool = False):
        x = bisect_left(dataset_values, canvas_value)
        y = np.linspace(5, self.stacked_canvas.shape[0], 10)

        return go.Scatter(
            x=[x] * len(y),
            y=y,
            mode='lines',  # You could also use 'lines+markers' if you want visible points
            line=dict(color='white', width=vw(0.15), dash="longdash"),
            showlegend=True,
            visible=visible,
            hovertemplate='Canvas origin<extra></extra>',  # Added y-value to hover
            name='canvas'
        )


class VisResolutions:
    """
    Visualize image resolutions in a dataset

    Args:
        image_stats: Output of Images.get_stats method
        layout_color: base color for the layout segments
    """
    def __init__(self, image_stats: pd.DataFrame, layout_color='#41946A'):
        self. image_stats = image_stats
        self.layout_color = layout_color

    def _prepare_data(self) -> pd.DataFrame:
        overall_stats = self.image_stats['overall']

        # preparing px data
        relevant_cols = ['height', 'width', 'mode']
        data = overall_stats[relevant_cols].value_counts().reset_index()
        data.columns = ['height', 'width', 'mode', 'count']

        return data

    @staticmethod
    def _get_midpoints(resolutions: dict) -> list:
        """Get range midpoints for annotations"""
        res_widths = [res['w'] for res in resolutions.values()]
        res_widths.append(0) # add origin at the end
        midpoints = [(res_widths[i] + res_widths[i + 1]) / 2
                     for i in range(len(res_widths) - 1)]

        return midpoints

    @staticmethod
    def _get_resolution_segment(row: pd.Series | dict, resolutions: dict) -> str:
        """Return segment name given the width and height"""
        for name, res in reversed(resolutions.items()):
            if row['width'] <= res['w'] and row['height'] <= res['h']:
                return name

    def _get_resolution_counts(self, data: pd.DataFrame, resolutions: dict) -> dict:
        """Value counts of each resolution segment"""
        get_segment = partial(self._get_resolution_segment, resolutions=resolutions)
        data['resolution'] = data.apply(get_segment, axis=1)

        counts = data.groupby('resolution')['count'].sum().to_dict()
        return counts

    def _get_layout_colormap(self, data: pd.DataFrame, resolutions: dict) -> dict:
        """Return colormap for the resolution segments weighed with image counts"""
        empty_factor = 0.05
        max_factor = 0.6

        resolution_counts = self._get_resolution_counts(data, resolutions)
        total = sum(resolution_counts.values())

        colormap = dict()
        color_empty = hex_to_rgba(self.layout_color, factor=empty_factor, alpha=0.95) # no images here
        for name, size in resolutions.items():
            # no images of that resolution
            if name not in resolution_counts:
                colormap[name] = color_empty
            else:
                # higher the ratio brighter the color
                factor = resolution_counts[name]/total * max_factor + empty_factor
                color = hex_to_rgba(self.layout_color, factor=factor, alpha=0.95)
                colormap[name] = color

        return colormap

    def _add_layout(self, fig: go.Figure, data: pd.DataFrame, resolutions: dict) -> go.Figure:
        """Display resolution segments"""
        names = resolutions.keys()
        sizes = resolutions.values()
        x_midpoints = self._get_midpoints(resolutions)
        resolution_counts = self._get_resolution_counts(data, resolutions)

        colormap = self._get_layout_colormap(data, resolutions)

        for name, size, x in zip(names, sizes, x_midpoints):
            w, h = tuple(size.values())
            perimeter_x = [0, 0, w, w]
            perimeter_y = [0, h, h, 0]

            if name in resolution_counts:
                text = f'{name}: {resolution_counts[name]}'
            else:
                text = f'{name}: 0'

            fig.add_trace(go.Scatter(x=perimeter_x,
                                     y=perimeter_y,
                                     mode='lines',
                                     fill='toself',
                                     fillcolor=colormap[name],
                                     hoverinfo='text',
                                     text=text,
                                     line=dict(color='gray', width=vw(0.02)),
                                     showlegend=False,
                                     name=name))

            fig.add_annotation(
                x=x,
                y=100,
                text=name,
                showarrow=False,
                font=AxisTick.font
            )

        return fig

    def _get_relevant_resolutions(self, data: pd.DataFrame) -> dict:
        """Filter relevant resolution segments"""
        # common resolutions
        resolutions = configs.COMMON_RESOLUTIONS

        # max height and width among images
        max_w = data['width'].max()
        max_h = data['height'].max()

        # highest resolution segment containing at least one image
        max_resolution = self._get_resolution_segment(dict(width=max_w, height=max_h), resolutions)
        res_max_w = resolutions[max_resolution]['w']
        res_max_h = resolutions[max_resolution]['h']

        # Filter resolutions that contain images (leave lower resolutions)
        relevant_resolutions = {
            name: res for name, res in resolutions.items()
            if res['w'] <= res_max_w or res['h'] <= res_max_h
        }

        return relevant_resolutions

    @staticmethod
    def _get_scatter_trace(data: pd.DataFrame, mode, color, symbol):
        mode_data = data[data['mode'] == mode]
        size_ratio = data['count'].max() / data['count'].mean()
        size_ref = 2.0 * size_ratio / (vw(0.07) ** 2)

        return go.Scatter(
                x=mode_data['width'],
                y=mode_data['height'],
                mode='markers',
                name=mode,
                customdata=[mode] * mode_data['count'].sum(),
                marker=dict(
                    line=dict(color=color, width=vw(0.05)),
                    symbol=symbol,
                    size=mode_data['count'],
                    sizeref=size_ref,
                    color=color,
                    opacity=0.7
                ),
                hovertemplate='%{x} x %{y}<br>'
                              '%{customdata}: %{marker.size}<br><br><extra></extra>',
                text=mode_data['resolution']
            )

    def plot(self):
        data = self._prepare_data()

        fig = go.Figure()

        resolutions = self._get_relevant_resolutions(data)
        fig = self._add_layout(fig, data, resolutions)

        mono_trace = self._get_scatter_trace(data=data, mode='mono', color='#3e3d39', symbol='square')
        rgb_trace = self._get_scatter_trace(data=data, mode='rgb', color='#ede492', symbol='diamond')

        fig.add_trace(mono_trace)
        fig.add_trace(rgb_trace)

        fig.update_layout(
            showlegend=True,
            width=vw(36),
            height=vh(34),
            plot_bgcolor=Color.Transparent,
            paper_bgcolor=Color.Transparent,
            margin=dict(r=0, l=0, t=0, b=0),
            hoverlabel=dict(font=Hover.font),
            legend=dict(title=None, font=Legend.font,
                        x=-0.13, y=0.92, orientation='v'),
        )

        fig.update_xaxes(
            showgrid=False,
            showline=False,
            tickfont=AxisTick.font,
            title_font=AxisTitle.font,
            tickvals=[0] + [v['w'] for v in resolutions.values()],
            ticklabelposition='inside'  # Move labels inside the axes
        )

        fig.update_yaxes(
            showgrid=False,
            showline=False,
            title_font=AxisTitle.font,
            tickfont=AxisTick.font,
            tickvals=[0] + [v['h'] for v in resolutions.values()],
            ticklabelposition='inside'  # Move labels inside the axes
        )

        st.plotly_chart(fig, use_container_width=True, config=PlotConfig.FULLSCREEN)



def plot_progress_pie(progress: int | float, total: int | float,
                      label: str='progress', color: str='red') -> None:
    """
    Visualize progress using pie chart

    Args:
        progress: A number representing the progress
        total: A number representing the max limit for progress
        label: String used as a hover for progress
        color: Color of progress
    """

    labels = [label, ' ']
    values = [progress, total - progress]

    fig = go.Figure([go.Pie(labels=labels,
                            values=values,
                            hole=0.95,
                            hoverinfo='label+value')])

    fig.update_traces(marker=dict(colors=[color, Color.Transparent],
                                  line=dict(color='#2F3543', width=vw(0.2)),
                                  pattern=dict(shape=['-'], size=vw(4))),
                      textposition='outside',
                      textinfo='text',
                      sort=False,
                      hoverlabel_font=Hover.font)

    # Update the layout of the chart
    fig.update_layout(
        height=vh(50),
        plot_bgcolor=Color.Transparent,
        paper_bgcolor=Color.Transparent,
        margin=dict(r=0, l=0, t=vh(0.2), b=vh(0.2)),
        annotations=[
            dict(
                text=str(progress),
                showarrow=False,
                font=dict(size=vw(1.8), color=Color.Secondary, style=Text.style),
                x=0.5,
                y=0.5,
                xanchor='center',
                yanchor='middle'
            )],
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True, config=PlotConfig.DEFAULT)


class VisOverlapRatios:
    """
    Visualizes overlap ratios for selected classes using a box plot

    Args:
        colormap: dict mapping class names to colors
        sample_size: max number of points to show on box chart
    """
    def __init__(self, colormap: dict, sample_size: int=5000):
        self.colormap = colormap
        self.sample_size = sample_size

    @st.cache_data(show_spinner=False)
    def plot(_self, overlap_stats: pd.DataFrame) -> None:
        """
        Plot the box chart

        Args:
            overlap_stats: Output of Overlaps.get_stats method
        """
        box_df = _self._prepare_data(overlap_stats)
        if box_df is None:
            return

        fig = px.box(box_df,
                     x="object", y="ratio",
                     points='all', color='object',
                     color_discrete_map=_self.colormap)

        fig.update_layout(
            autosize=False,
            plot_bgcolor=Color.Transparent,
            paper_bgcolor=Color.Transparent,
            width=vw(35),
            height=vh(33),
            font=dict(size=vw(1), family='Arial'),
            showlegend=False,
            xaxis_title=None,
            yaxis_title='Overlap ratio',
            hoverlabel=dict(font=Hover.font),
            margin=dict(r=0, l=0, t=0, b=0))

        fig.update_xaxes(**_self._axis_params())
        fig.update_yaxes(**_self._axis_params())

        fig.update_traces(marker_line_width=vw(0.05),
                          marker_line_color="black")

        st.plotly_chart(fig, use_container_width=True, config=PlotConfig.DEFAULT)

    def _prepare_data(self, overlap_stats: pd.DataFrame) -> pd.DataFrame | None:
        # filter positive overlaps
        positive_overlap_indices = overlap_stats['overlap_size'] > 0
        positive_overlaps = overlap_stats[positive_overlap_indices]

        n_positives = positive_overlaps.shape[0]
        if n_positives == 0:
            self._render_empty()
            return

        # handle 'All' case
        for col in ['relate', 'relate_with']:
            if positive_overlaps[col].nunique() > 1:
                positive_overlaps.loc[:, col] = 'All'

        object1_df = positive_overlaps[['relate', 'relate_size', 'overlap_size']]
        object2_df = positive_overlaps[['relate_with', 'with_size', 'overlap_size']]

        # combine both forward and reverse relations into a single dataframe
        object1_df.columns = object2_df.columns = ['object', 'size', 'overlap_size']
        df_combined = pd.concat([object1_df, object2_df], axis=0)

        df_combined['ratio'] = df_combined['overlap_size'] / df_combined['size'] * 100
        if df_combined.shape[0] > self.sample_size:
            df_combined = df_combined.sample(self.sample_size).sort_values('object', ascending=False)

        return df_combined

    @staticmethod
    def _axis_params() -> dict:
        return dict(showgrid=False,
                    title_standoff=vw(0.8),
                    title_font=AxisTitle.font,
                    tickfont=AxisTick.font)

    @staticmethod
    def _render_empty() -> None:
        """Handle the case of no positive overlaps"""
        fig = go.Figure()

        fig.update_layout(
            plot_bgcolor=Color.Transparent,
            paper_bgcolor=Color.Transparent,
            annotations=[
                dict(x=2.6, y=2.5, text="NO POSITIVE OVERLAPS", showarrow=False,
                     font=dict(size=vw(1.5), color=Color.Secondary, weight='bold',
                               family=Title.family)),
            ],
            xaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False)
        )

        st.plotly_chart(fig, use_container_width=True, config=PlotConfig.DEFAULT)


class VisOverlap:
    """
    Visualizes overlaps between two classes with clustered case scenarios

    It creates interactive visualizations of overlap statistics between two
    classes, represented as circles and shows real examples using images
    from the dataset

    Args:
        overlap_stats: Output of Overlaps.get_stats method
        matching_dict: Dictionary matching label and image filenames
        circle1_name: Name of the first circle (class)
        circle2_name: Name of the second circle (class)
        colormap: Dictionary mapping class names to colors
        n_cases: Number of overlap cases to visualize
        n_images: Number of images to display for each case
        min_samples: Minimum number of samples required for clustering
    """
    class OverlapCircle:
        """
        A nested class representing a circle in the overlap visualization.

        Args:
            name: name of the circle

        Methods:
            r: calculates the radius of the circle
            perimeter: returns n points on circle's perimeter
        """
        def __init__(self, name: str):
            self.name = name
            self.x = 0
            self.y = 0

            self.size = None
            self.overlap = None

        @property
        def r(self) -> float | int:
            if self.size:
                return np.sqrt(self.size / np.pi)

        @property
        def perimeter(self, n: int=100) -> tuple:
            theta = np.linspace(0, int(2 * np.pi + 1), n)

            perimeter_x = self.x + self.r * np.cos(theta)
            perimeter_y = self.y + self.r * np.sin(theta)

            return perimeter_x, perimeter_y

    def __init__(self,
                 overlap_stats: pd.DataFrame,
                 matching_dict: dict,
                 circle1_name: str,
                 circle2_name: str,
                 colormap: dict,
                 n_cases: int,
                 n_images: int,
                 min_samples: int=10):

        self.overlap_stats = overlap_stats
        self.matching_dict = matching_dict
        self.colormap = colormap

        self.circle1_name = circle1_name
        self.circle2_name = circle2_name

        self.circle1 = self.OverlapCircle(name=circle1_name)
        self.circle2 = self.OverlapCircle(name=circle2_name)

        self.n_cases = n_cases
        self.n_images = n_images
        self.min_samples = min_samples

    def _get_kmeans_labels(self) -> np.ndarray | list:
        """
        Perform K-means clustering on overlap data and return cluster labels.
        """
        n_clusters = self.n_cases - 1 # no overlap case is not a cluster

        positive_overlap_indices = self.overlap_stats['overlap_size'] > 0
        n_positive_overlaps = sum(positive_overlap_indices)

        if n_positive_overlaps == 0:
            st.toast(f'At least {self.min_samples} overlaps needed for clustering',
                     icon=':material/info:')
            return 0 # only no-overlap cluster

        elif n_positive_overlaps < self.min_samples:
            st.toast(f'At least {self.min_samples} overlaps needed for clustering',
                     icon=':material/info:')
            return 1 # one cluster for positive overlaps

        positive_overlaps = self.overlap_stats[positive_overlap_indices]
        positive_overlaps = positive_overlaps.fillna(0)  # in case of 0/0 division

        relate_overlap_ratio = positive_overlaps.overlap_size / positive_overlaps.relate_size
        with_overlap_ratio = positive_overlaps.overlap_size / positive_overlaps.with_size

        relate_overlap_ratio = relate_overlap_ratio.values.reshape(-1,1)
        with_overlap_ratio = with_overlap_ratio.values.reshape(-1,1)

        if self.circle1.name == self.circle2.name:
            clustering_data = relate_overlap_ratio
            centroids_init = np.linspace(0, 1, n_clusters).reshape(-1, 1)
        else:
            clustering_data = np.hstack((relate_overlap_ratio, with_overlap_ratio))
            centroids_init = generate_proportional_centroids(n_clusters)


        km = KMeans(n_clusters=n_clusters, init=centroids_init)
        km.fit(clustering_data)

        return np.array(km.labels_) + 1

    def _get_pairs(self, case_df: pd.DataFrame) -> pd.DataFrame | None:
        """Get label-image pairs"""
        matched_filename_indices = case_df['filename'].isin(self.matching_dict.keys())
        # no matched indices
        if not any(matched_filename_indices):
            return
        selected_pairs = case_df[matched_filename_indices].sample(self.n_images, replace=True)

        return selected_pairs

    def _calculate_circle_properties(self, case_df: pd.DataFrame) -> None:
        self.circle1.size = np.median(case_df['relate_size'])
        self.circle2.size = np.median(case_df['with_size'])
        overlap_size = np.median(case_df['overlap_size'])

        distance = calculate_circles_distance(r1=self.circle1.r, r2=self.circle2.r, a=overlap_size)
        self.circle2.x = distance

        self.circle1.overlap = math.ceil(overlap_size / self.circle1.size * 100)
        self.circle2.overlap = math.ceil(overlap_size / self.circle2.size * 100)

    def _get_circle_traces(self, case_df: pd.DataFrame) -> list:
        self._calculate_circle_properties(case_df)

        overlap_text = self._get_overlap_text()

        circle_scatter = partial(self._fill_scatter, fill_alpha=0.6, fill_factor=0.6, show_legend=False)
        trace_circle1 = circle_scatter(perimeter=self.circle1.perimeter, name=self.circle1.name)
        trace_circle2 = circle_scatter(perimeter=self.circle2.perimeter, name=self.circle2.name)

        trace_overlap = go.Scatter(x=[self.circle2.x/2],
                                   y=[0],
                                   opacity=0,
                                   hoverinfo='text',
                                   text=overlap_text,
                                   visible=False,
                                   showlegend=False,
                                   marker=dict(size=50))

        circle_traces = [trace_circle1, trace_circle2]
        if self.circle1.size < self.circle2.size:
            circle_traces = circle_traces[::-1]

        circle_traces.append(trace_overlap)

        return circle_traces

    @staticmethod
    def _get_empty_trace():
        """Create a placeholder empty trace"""
        return go.Scatter(x=[None], y=[None], showlegend=False, visible=False)

    @staticmethod
    def _get_no_image_trace():
        img = cv2.imread('assets/other/no_image.png')
        trace = px.imshow(img).data[0]
        trace.update(visible=False, hovertemplate=None, hoverinfo='skip')

        return trace

    def _get_image_traces(self, pair: pd.Series | None) -> tuple:
        if pair is None:
            return self._get_no_image_trace(), [self._get_empty_trace()] * 2

        img = cv2.imread(pair.image_filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped_img, pair = crop_aggregate_box(img, pair, square=pair.overlap_size == 0, padding=0.1)

        image_trace = px.imshow(cropped_img).data[0]
        image_trace.update(visible=False, hovertemplate=None, hoverinfo='skip')

        bbox1 = get_rect_vertices(cropped_img, pair.relate_x, pair.relate_y,
                                  pair.relate_w, pair.relate_h)

        bbox2 = get_rect_vertices(cropped_img, pair.with_x, pair.with_y,
                                  pair.with_w, pair.with_h)

        bbox_scatter = partial(self._fill_scatter, fill_alpha=0.25, fill_factor=0.5)
        bbox1_trace = bbox_scatter(perimeter=bbox1, name=self.circle1.name)
        bbox2_trace = bbox_scatter(perimeter=bbox2, name=self.circle2.name)

        bbox_traces = [bbox1_trace, bbox2_trace]
        if self.circle1.size < self.circle2.size:
            bbox_traces = bbox_traces[::-1]

        return image_trace, bbox_traces

    def _generate_case_traces(self, case: str) -> tuple:
        relevant_indices = self.overlap_stats['case'] == case
        case_df = self.overlap_stats[relevant_indices]

        selected_pairs = self._get_pairs(case_df)
        circle_traces = self._get_circle_traces(case_df=case_df)

        image_traces = []
        # no matches between labels and images
        if selected_pairs is None:
            image_traces.extend([self._get_image_traces(pair=None)] * self.n_images)

        else:
            # found matches between images and labels
            for (_, pair) in selected_pairs.iterrows():
                image_traces.append(self._get_image_traces(pair=pair))

        return circle_traces, image_traces

    def plot(self):
        """
        Create and display the overlap and image charts with slider controls
        """
        fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6])

        n_overlaps = self.overlap_stats.shape[0]
        # initializing with zeros
        self.overlap_stats['case'] = np.zeros(n_overlaps)

        # case cluster labels
        case_clusters = self._get_kmeans_labels()

        # assign clusters to positive overlaps
        positive_overlap_indices = self.overlap_stats['overlap_size'] > 0
        self.overlap_stats.loc[positive_overlap_indices, 'case'] = case_clusters

        case_counts = self.overlap_stats['case'].value_counts()
        slider_range = case_counts.index.tolist()

        for case in slider_range:
            circle_traces, image_traces = self._generate_case_traces(case)
            fig.add_traces(circle_traces, rows=[1, 1, 1], cols=[1, 1, 1])

            for traces in image_traces:
                image_trace, bbox_traces = traces

                fig.add_trace(image_trace, row=1, col=2)
                fig.add_traces(bbox_traces, rows=[1, 1], cols=[2, 2])

            # annotations
            overlap_header = self._overlap_header_annotation(case=case)
            fig.add_annotation(overlap_header)

        n_traces_per_case = 6

        # Set the first set of traces to visible
        for i in range(n_traces_per_case):
            fig.data[i].update(visible=True)

        # Set the first set of annotations to visible
        fig.layout.annotations[0].update(visible=True)

        steps = []
        image_steps = []

        labels_slider = [f'case {i}' for i in range(1, self.n_cases + 1)]
        images_slider = []
        for i in labels_slider:
            images_slider.append(i)
            images_slider.extend([''] * (self.n_images - 1))

        for i in range(len(slider_range)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"annotations": []}],
                label=labels_slider[i]
            )

            traces_start_index = i * (3 + self.n_images * 3)  # where case traces start
            for k in range(traces_start_index, traces_start_index + n_traces_per_case):
                step["args"][0]["visible"][k] = True

            annotation = deepcopy(fig.layout.annotations[i])
            annotation.visible = True

            step["args"][1]["annotations"] = [annotation]
            steps.append(step)

            for j in range(self.n_images):
                image_step = dict(
                    method="update",
                    args=[{"visible": [False] * len(fig.data)},
                          {"annotations": []}],
                    label=images_slider[i * self.n_images + j]
                )

                traces_start_index = i * (3 + self.n_images * 3)
                image_step["args"][0]["visible"][traces_start_index] = True
                image_step["args"][0]["visible"][traces_start_index + 1] = True
                image_step["args"][0]["visible"][traces_start_index + 2] = True
                image_step["args"][0]["visible"][traces_start_index + 3 + 3 * j] = True
                image_step["args"][0]["visible"][traces_start_index + 4 + 3 * j] = True
                image_step["args"][0]["visible"][traces_start_index + 5 + 3 * j] = True

                image_step["args"][1]["annotations"] = [annotation]

                image_steps.append(image_step)

        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "Overlap: "},
                pad={"t": 80},
                steps=steps,
                ticklen=0,
                tickcolor="black",
                font=dict(color=Color.Secondary, size=18),
                x=0.02,
                y=0.2,
                len=0.34,
                bordercolor=Color.Secondary,
                bgcolor="black",
            ),

            dict(
                active=0,
                currentvalue={"prefix": ""},
                pad={"t": 80},
                steps=image_steps,
                ticklen=3,
                tickcolor="white",
                font=dict(color=Color.Secondary, size=18),
                x=0.48,
                y=0.2,
                len=0.5,
                bordercolor=Color.Secondary,
                bgcolor="black",
            )
        ]

        fig.update_layout(
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.43,
                xanchor="center",
                x=0.43,
                font=Legend.font
            ),
            sliders=sliders,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=True,
            plot_bgcolor=Color.Transparent,
            paper_bgcolor=Color.Transparent,
            margin=dict(t=0, b=vh(10)),
            height=vh(37),
            hoverlabel=dict(font=Hover.font),
            font=Text.font
        )

        fig.update_traces(hoverlabel_font_size=18)

        fig.update_xaxes(showticklabels=False, showgrid=False, showline=False,
                         zeroline=False, ticks='')
        fig.update_yaxes(showticklabels=False, showgrid=False, showline=False,
                         zeroline=False, ticks='')

        st.plotly_chart(fig, use_container_width=True, config=PlotConfig.DEFAULT)

    def _fill_scatter(self, perimeter, name, fill_alpha, fill_factor, show_legend=True):
        """
         Create a filled scatter plot trace for a circle.

         :param perimeter: Tuple of x and y coordinates for the circle's perimeter
         :param name: Name of the circle (class)
         :param fill_alpha: Alpha value for the fill color
         :param fill_factor: Factor to adjust the fill color intensity
         :param show_legend: Boolean to control legend visibility
         :return: Plotly Scatter trace object
         """
        perimeter_x, perimeter_y = perimeter

        line_color = hex_to_rgba(self.colormap[name], alpha=1, factor=1)
        fillcolor = hex_to_rgba(self.colormap[name], alpha=fill_alpha, factor=fill_factor)

        return go.Scatter(x=perimeter_x,
                          y=perimeter_y,
                          mode='lines',
                          fill='toself',
                          fillcolor=fillcolor,
                          hoverinfo='text',
                          text=name,
                          line=dict(color=line_color, width=2),
                          visible=False,
                          showlegend=show_legend,
                          name=name)

    def _overlap_header_annotation(self, case: str) -> dict:
        """Generates an annotation to display the general case statistics"""
        n_total = self.overlap_stats.shape[0]

        case_indices = self.overlap_stats['case'] == case
        case_instances = self.overlap_stats[case_indices]
        n_case = case_instances.shape[0]

        case_instances_ratio = n_case / n_total * 100
        if case_instances_ratio < 0.1:
            text = f'{n_case} relations (< 0.1%)'
        else:
            text = f'{n_case} relations ({case_instances_ratio:.1f}%)'

        annotation = dict(x=self.circle2.x / 2,
                          y=max(self.circle1.r, self.circle2.r) + 0.02,
                          visible=False,
                          showarrow=False,
                          font=dict(color="white", size=vw(1)),
                          text=text)

        return annotation

    def _get_overlap_text(self) -> str:
        # Define styling constants
        _styles = {
            'title': {
                'font-weight': 'bold',
                'font-size': '25px'
            },
            'label': {
                'font-weight': 'bold',
                'font-size': '18px'
            }
        }

        def create_style_string(styles):
            return ';'.join(f'{key}:{value}' for key, value in styles.items())

        # Create the HTML structure using a template
        template = """
            <span style='{title_style}'>Overlap</span>
            <br><br><span style='{label_style};color:{color1}'>{name1}</span>: {overlap1:.3g}%
            <br><span style='{label_style};color:{color2}'>{name2}</span>: {overlap2:.3g}%
        """

        # Format the template with all values
        overlap_text = template.format(
            title_style=create_style_string(_styles['title']),
            label_style=create_style_string(_styles['label']),
            color1=self.colormap[self.circle1.name],
            name1=self.circle1.name.capitalize(),
            overlap1=self.circle1.overlap,
            color2=self.colormap[self.circle2.name],
            name2=self.circle2.name.capitalize(),
            overlap2=self.circle2.overlap
        )

        # Remove extra whitespace and newlines
        return ' '.join(overlap_text.split())


def plot_overlap_pie(labels: list, values: list,  height: float,
                     colors: list=None, title: str=None,
                     display_counts: bool=True) -> None:
    """
    Plot a pie chart in Overlaps page style

    Args:
        labels: labels list to pass to go.Pie
        values: values list to pass to go.Pie
        colors: colors list to pass to go.Pie.marker
        title: optional string to display below the chart
        height: height of the chart as a fraction of VH
        display_counts: whether to display counts or percentages

    """

    total = sum(values)
    text = [value if value > 0 else '' for value in values]
    hover_info = 'label + percent'

    if not display_counts:
        percentages = [(value / total) * 100 for value in values]
        text = [f'{percentage:.3g}%' if percentage > 0 else '' for percentage in percentages]
        hover_info = 'label + value'

    fig = go.Figure([go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=dict(color=Color.Background, width=vw(0.15))),
        textfont=dict(size=vw(1.1), style=Text.style),
        hoverinfo=hover_info,  # Display label and percentage on hover
        textinfo='text',  # Display actual values on the pie chart
        text=text,
        hole=.3,
        sort=False,
        opacity=0.9,
        pull=0.03
    )])

    fig.update_layout(
        # Set the background color to fully transparent
        plot_bgcolor=Color.Transparent,
        paper_bgcolor=Color.Transparent,
        hoverlabel=dict(font=Hover.font),
        autosize=False,
        height=height,
        margin=go.layout.Margin(
            l=vw(2),
            r=vw(2),
            b=vw(5),
            t=vh(2),
            pad=0
        ),
        showlegend=False
    )

    if title:
        fig.add_annotation(
            text=title,
            x=0.5,
            y=-0.2,  # Position below the chart
            showarrow=False,
            font=dict(size=vw(0.9), color=Color.Primary,
                      family=Text.family, weight='bold'),
            xanchor='center',
            yanchor='top',
            xref="paper",
            yref="paper",
            visible=True
        )

    st.plotly_chart(fig, use_container_width=True, config=PlotConfig.DEFAULT)
