import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data with type handling
@st.cache_data
def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    df.drop("customerID", axis=1, inplace=True)
    return df

# Helper: Apply white text to axes
def set_white_text(ax):
    ax.title.set_color("white")
    ax.title.set_fontsize(16)
    ax.xaxis.label.set_color("white")
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_color("white")
    ax.yaxis.label.set_size(14)
    ax.tick_params(colors="white", labelsize=12)
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color("white")
            text.set_fontsize(12)

def run_eda():
    st.markdown("<h1 style='text-align: center;'>📊 Telco Customer Churn - Exploratory Data Analysis</h1>", unsafe_allow_html=True)

    df = load_data()

    # 📌 Overview
    st.subheader("📌 Dataset Overview")
    st.dataframe(df.head())

    if st.checkbox("Show Summary Statistics"):
        st.write(df.describe(include='all'))

    # 📈 Churn distribution
    st.subheader("📈 Churn Distribution")
    churn_data = df['Churn'].value_counts()
    fig, ax = plt.subplots(facecolor='none', figsize=(14, 6))
    ax.set_facecolor('none')
    wedges, texts, autotexts = ax.pie(
        churn_data,
        labels=churn_data.index,
        autopct='%1.1f%%',
        colors=sns.color_palette(["#6D8DA5","#47708F", "#CA7A5A",  "#D6B340", "#85BD44", ]),
        startangle=90
    )
    for text in texts:
        text.set_color("white")
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(12)
    ax.axis('equal')
    st.pyplot(fig)

    # 🔍 Column selection
    st.subheader("🔍 Column Type Selection")
    column_type = st.radio("Select column type for EDA:", ["Numerical", "Categorical"])

    if column_type == "Numerical":
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        num_cols = [col for col in num_cols if col != 'Churn']
        selected_num = st.multiselect("Select numerical columns to analyze:", num_cols)

        for col in selected_num:
            with st.expander(f"📊 Distribution and Comparisons for {col}", expanded=False):
                st.write(f"#### Histogram + KDE of {col}")
                fig, ax = plt.subplots(facecolor='none', figsize=(14, 6))
                ax.set_facecolor('none')
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color="skyblue", stat="density", kde_kws={"bw_adjust": 0.8})
                set_white_text(ax)
                sns.despine()
                plt.tight_layout(pad=0.6)
                st.pyplot(fig)

                st.write(f"#### Boxplot of {col} by Churn")
                fig, ax = plt.subplots(facecolor='none', figsize=(14, 6))  # Reduced size too
                ax.set_facecolor('none')
                sns.boxplot(x='Churn', y=col, data=df, ax=ax, palette="coolwarm", width=0.3)  # width changed
                set_white_text(ax)
                sns.despine()
                plt.tight_layout(pad=0.6)
                st.pyplot(fig)


                st.write(f"#### Violin Plot of {col} by Churn")
                fig, ax = plt.subplots(facecolor='none', figsize=(14, 6))  # Reduced overall figure size too
                ax.set_facecolor('none')
                sns.violinplot(x='Churn', y=col, data=df, ax=ax, palette="Set2", width=0.4)  # Adjust width here
                set_white_text(ax)
                sns.despine()
                plt.tight_layout(pad=0.6)
                st.pyplot(fig)

    elif column_type == "Categorical":
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        cat_cols = [col for col in cat_cols if col not in ['customerID', 'Churn']]
        selected_cat = st.multiselect("Select categorical columns to analyze:", cat_cols)

        for col in selected_cat:
            with st.expander(f"📊 Count and Distribution of {col}", expanded=False):
                st.write(f"#### Countplot of {col}")
                fig, ax = plt.subplots(facecolor='none', figsize=(14, 6))
                ax.set_facecolor('none')
                plot = sns.countplot(x=col, data=df, ax=ax, palette="Set2")

                # Slim and center bars
                for p in plot.patches:
                    original_width = p.get_width()
                    new_width = 0.3
                    diff = original_width - new_width
                    p.set_width(new_width)
                    p.set_x(p.get_x() + diff / 2)

                set_white_text(ax)
                ax.tick_params(axis='x', rotation=0)
                sns.despine()
                plt.tight_layout(pad=0.6)
                st.pyplot(fig)



                st.write(f"#### {col} Distribution by Churn")
                fig, ax = plt.subplots(facecolor='none', figsize=(14, 6))
                ax.set_facecolor('none')
                plot = sns.countplot(x=col, hue='Churn', data=df, ax=ax, palette="coolwarm", dodge=True)

                # Slim and center bars
                for p in plot.patches:
                    original_width = p.get_width()
                    new_width = 0.4
                    diff = original_width - new_width
                    p.set_width(new_width)
                    p.set_x(p.get_x() + diff / 2)

                set_white_text(ax)
                ax.tick_params(axis='x', rotation=0)
                sns.despine()
                plt.tight_layout(pad=0.6)
                st.pyplot(fig)


                st.write(f"#### Pie Chart of {col}")
                pie_data = df[col].value_counts()
                fig, ax = plt.subplots(facecolor='none', figsize=(14, 6))
                ax.set_facecolor('none')
                wedges, texts, autotexts = ax.pie(
                    pie_data,
                    labels=pie_data.index,
                    autopct='%1.1f%%',
                    colors=sns.color_palette(["#CA7A5A",  "#D6B340", "#5B91BD", "#85BD44", "#1E759E"]),
                    startangle=90
                )
                for text in texts + autotexts:
                    text.set_color("white")
                    text.set_fontsize(12)
                ax.axis('equal')
                st.pyplot(fig)
