# Deeva 🚀   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  [![PyPI](https://img.shields.io/pypi/v/deeva.svg)](https://pypi.org/project/deeva/) [![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![GitHub Repo stars](https://img.shields.io/github/stars/vbyan/deeva?style=social)](https://github.com/vbyan/deeva)

**Your Smart Analytics Companion for Object Detection Datasets**

<br>

## 🎯 Overview

**Deeva** is a powerful yet easy-to-use analytics toolkit that makes exploring **Object Detection** datasets a breeze, whether you're just starting out or a seasoned pro. 

Built with **Streamlit**, it offers an intuitive interface packed with features that let you dive into your data quickly or take a deeper look when you need it.
Deeva is designed to simplify data exploration and reporting, so you can get meaningful insights without the hassle.

### Key Features

- **💻 Run locally**: Launch effortlessly on your **_local machine_** for seamless, offline use.
- **🚀 Instant Setup**: Quickly start visualizing data by pointing Deeva to a specific dataset folder.
- **📊 Rich Interactive Dashboards**: Build insightful, interactive dashboards for rich data exploration with minimal effort.
- **🎨 Customizable CLI**: Use simple command-line commands to launch Deeva with flexible paths and configurations.
- **💾 Smart Caching**: Efficient processing with intelligent data caching for large datasets
- **🎲 Built-in Toy Datasets**: Quickly get started with the included `coco128` dataset, perfect for initial experimentation.

<br>

## 🛠 Installation

install with **pip**:

```bash
$ pip install deeva
```

Alternatively, use a **virtual environment** (recommended):

```bash
$ python3 -m venv myenv
$ source myenv/bin/activate

$ pip install deeva
```

## ⚡ Quickstart
After installation, launch **Deeva** by running:

```bash
$ deeva start
```

This will open the **input page** where you can specify the **data path**.

&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://raw.githubusercontent.com/vbyan/deeva/main/assets/input_page.gif"></img>

### Data structure

Your dataset **folder** should look like this:

```plaintext
data-path/
├── images/        # Folder containing image files (e.g., .jpg, .png)
├── labels/        # Folder containing label files (e.g., .txt, .xml)
└── labelmap.txt   # A file mapping class IDs to class labels (optional)
```
<br>

## 💡 Insights & Analytics

Deeva offers a powerful set of **statistical insights** to give you a detailed understanding of your dataset, including:

### 1. **File Matching and Integrity**

<br> &nbsp;&nbsp;&nbsp;&nbsp;<img src="https://raw.githubusercontent.com/vbyan/deeva/main/assets/data_match.gif">

- **Image-Label Matching**: Calculates how many images have **corresponding labels** (and vice versa).
- **Filename Consistency**: Identifies **misaligned or corrupted files** in images and labels.
- **Data Cleaning**: Provides tools to **identify and isolate** mismatched or corrupted files.


### 2. **Dataset Overview**

<br> &nbsp;&nbsp;&nbsp;&nbsp;<img src="https://raw.githubusercontent.com/vbyan/deeva/main/assets/overall.gif">

- **File Formats & Backgrounds**: View format distribution (`yolo` vs. `voc`, `jpeg` vs. `png`).
- **Class Distribution**: Displays instance counts and images per class, highlighting any **class imbalances**.
- **Class Co-occurrence**: Shows how frequently different classes **appear together**.


### 3. **Annotation Insights**

<br> &nbsp;&nbsp;&nbsp;&nbsp;<img src="https://raw.githubusercontent.com/vbyan/deeva/main/assets/annotations.gif">

- **Bounding Box Analysis**: Provides insights into box center, width/height, and median box sizes.
- **Box Size Distribution**: Analyzes box size categories with **adjustable thresholds** for small, medium, and large sizes.


### 4. **Image Statistics**

<br> &nbsp;&nbsp;&nbsp;&nbsp;<img src="https://raw.githubusercontent.com/vbyan/deeva/main/assets/images.gif">

- **Color Analysis**: Displays **dominant colors** and their tones extracted from images.
- **Image Dimensions**: Examines **height, width,** and aspect ratios across your dataset.
- **CBS (Contrast, Brightness, Saturation)**: Shows **contrast, brightness, and saturation** distributions across the dataset.


### 5. **Overlap Statistics**

<br> &nbsp;&nbsp;&nbsp;&nbsp;<img src="https://raw.githubusercontent.com/vbyan/deeva/main/assets/overlaps.gif">

- **Cases**: Classify and **cluster** overlapping instances from two specific classes into `n` predefined cases. Display representative **example images** for each case to help visualize typical overlap patterns.
- **Ratios**: Calculate and visualize the overlap ratio distributions for each class
- **With/without overlaps**: Present a side-by-side comparison of images and co-occurrences with and without overlaps

<br>

## 🔖 **Caching & Version control**

Deeva employs efficient **caching** to streamline your data processing workflow. For large datasets, users have the option to **sample a subset of the data**—allowing for quicker initial exploration. 

Data extracted during time-consuming operations can be saved as a **dataframe on disk** for effortless access in future sessions, enabling a faster, more efficient experience by skipping redundant processing steps.

To **track different versions** of your dataset you need to simply put them into different folders and **Deeva** will do the rest

<br>

## 🌟 **Contributing**

**Deeva** welcomes contributions! If you have ideas or want to add new features, please feel free to open a pull request or start a discussion on **GitHub**. 

<br>

## License

Deeva is completely free and open-source and licensed under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.



