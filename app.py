from controller import normalization as norm

import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# streamlit run app.py

PATH_FILES = 'C:/Users/Develop/Documents/Python/proyects/project_18/files/'
list_files = norm.import_list_csv(PATH_FILES)
df_names = list(list_files.keys())

df_contract = list_files[df_names[0]]
df_contract['EndDate'] = df_contract['EndDate'].replace('No', norm.np.nan)

list_files[df_names[0]]['EndDate'] = norm.adjust_data_time(
    df_contract['EndDate'], format_time='%Y-%m-%d %H:%M:%S')
list_files[df_names[0]]['BeginDate'] = norm.adjust_data_time(
    list_files[df_names[0]]['BeginDate'], format_time='%Y-%m-%d')
list_files[df_names[0]]['PaperlessBilling'] = list_files[df_names[0]
                                                         ]['PaperlessBilling'].apply(lambda x: 1 if x == 'Yes' else 0)
list_files[df_names[0]]['TotalCharges'] = norm.pd.to_numeric(
    list_files[df_names[0]]['TotalCharges'], errors='coerce').fillna(0)
for column in list_files[df_names[1]].columns:
    if column not in ['customerID', 'InternetService']:
        if list_files[df_names[1]][column].dtype == 'object':
            list_files[df_names[1]][column] = list_files[df_names[1]][column] \
                .apply(lambda x: 1 if x == 'Yes' else 0)
list_files[df_names[2]]['gender'] = list_files[df_names[2]
                                               ]['gender'].apply(lambda x: 1 if x == 'Male' else 0)
list_files[df_names[2]]['Partner'] = list_files[df_names[2]
                                                ]['Partner'].apply(lambda x: 1 if x == 'Yes' else 0)
list_files[df_names[2]]['Dependents'] = list_files[df_names[2]
                                                   ]['Dependents'].apply(lambda x: 1 if x == 'Yes' else 0)
list_files[df_names[3]]['MultipleLines'] = list_files[df_names[3]
                                                      ]['MultipleLines'].apply(lambda x: 1 if x == 'Yes' else 0).astype(bool)

main_df = list_files[df_names[0]].copy()
list_files.pop(df_names[0])
merge_files = list(list_files.values())


# -------------------------------
# 1. CONFIG & TITLE
# -------------------------------
st.set_page_config(page_title="Telco Customer Analysis", layout="wide")
st.title("📊 Telco Customer Churn - General Analysis Dashboard")
st.markdown("---")


@st.cache_data
def web_cache(df):
    return df


df_data = web_cache(norm.standarize_titles(
    norm.merge_list_dataframe(main_df, merge_files, 'customerID')))


df_data['timeservicedays'] = (
    df_data['enddate'] - df_data['begindate']).dt.days
df_data['objective'] = df_data['enddate'].apply(
    lambda x: 0 if norm.pd.notnull(x) else 1)
df_data['beginmonth'] = df_data['begindate'].dt.month
df_data['beginyear'] = df_data['begindate'].dt.year
# -------------------------------
# 3. BIG 3×2 PLOTLY DASHBOARD (your code, slightly cleaned)
# -------------------------------
# Ensure correct column names you used in notebook
col_map = {
    'contract': 'type',
    'paperlessbilling': 'paperlessbilling',
    'paymentmethod': 'paymentmethod',
    'onlinesecurity': 'onlinesecurity',
    'onlinebackup': 'onlinebackup',
    'deviceprotection': 'deviceprotection',
    'techsupport': 'techsupport',
    'streamingtv': 'streamingtv',
    'streamingmovies': 'streamingmovies',
    'objective': 'objective'  # assuming Churn = Yes/No → objective 1/0
}

df = df_data.copy()
# ----------------------------- SIDEBAR FILTERS -----------------------------
st.sidebar.header("🔍 Filters")
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df['type'].unique(),
    default=df['type'].unique()
)
payment_filter = st.sidebar.multiselect(
    "Payment Method",
    options=df['paymentmethod'].unique(),
    default=df['paymentmethod'].unique()
)

status_filter = st.sidebar.multiselect(
    "user Status",
    options=df['objective'].unique(),
    default=df['objective'].unique()
)

year_filter = st.sidebar.multiselect(
    "Year Status",
    options=df['beginyear'].sort_values(ascending=True).unique(),
    default=df['beginyear'].unique()
)

month_filter = st.sidebar.multiselect(
    "Month Status",
    options=df['beginmonth'].sort_values(ascending=True).unique(),
    default=df['beginmonth'].unique()
)

df_filtered = df[df['type'].isin(contract_filter) &
                 df['paymentmethod'].isin(payment_filter) &
                 df['objective'].isin(status_filter)]

# ----------------------------- MAIN DASHBOARD (Your 3x2 Plotly) -----------------------------
st.subheader("1. General Overview Dashboard")

fig = make_subplots(
    rows=3, cols=2,
    specs=[
        [{"type": "pie"}, {"type": "bar"}],
        [{"type": "pie"}, {"type": "bar"}],
        [{"type": "bar", "colspan": 2}, None]
    ],
    subplot_titles=(
        "Paperless Billing", "Payment Method Distribution",
        "Contract Type", "Add-on Services Usage",
        "Customers by Payment Method × Contract Type"
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# 1. Paperless Billing Pie
fig.add_trace(go.Pie(
    labels=['Yes', 'No'],
    values=df_filtered['paperlessbilling'].value_counts(),
    name="Paperless Billing",
    hole=0.4,
    marker_colors=['#FF6B6B', '#4ECDC4'],
    showlegend=False
), row=1, col=1)

# 2. Contract Type Pie
fig.add_trace(go.Pie(
    labels=df_filtered['type'].value_counts().index,
    values=df_filtered['type'].value_counts().values,
    name="Contract Type",
    hole=0.4,
    marker_colors=['#95E1D3', '#FCE38A', '#F38181'],
    showlegend=False
), row=2, col=1)

# 3. Payment Method Horizontal Bar
pm = df_filtered['paymentmethod'].value_counts()
fig.add_trace(go.Bar(
    y=pm.index,
    x=pm.values,
    orientation='h',
    name="Payment Methods",
    marker_color='#A8DADC',
    showlegend=False
), row=1, col=2)

# 4. Add-on Services Horizontal Bar
services = ['onlinesecurity', 'onlinebackup', 'deviceprotection',
            'techsupport', 'streamingtv', 'streamingmovies']
service_counts = df_filtered[services].apply(lambda x: (x == 'Yes').sum())
fig.add_trace(go.Bar(
    y=[s.replace("online", "Online ").replace(
        "streaming", "Streaming ").title() for s in service_counts.index],
    x=service_counts.values,
    orientation='h',
    name="Add-on Services",
    marker_color='#E63946',
    showlegend=False
), row=2, col=2)

# 5. Grouped Bar: Payment Method × Contract Type
dft = df_filtered.groupby(['paymentmethod', 'type']
                          ).size().reset_index(name='count')
for contract in dft['type'].unique():
    subset = dft[dft['type'] == contract]
    fig.add_trace(go.Bar(
        x=subset['paymentmethod'],
        y=subset['count'],
        name=contract,
        text=subset['count'],
        textposition='outside'
    ), row=3, col=1)

fig.update_layout(
    height=1000,
    barmode='group',
    title_text="Telco Customer General Analysis",
    legend_title="Contract Type",
    font=dict(size=12)
)

st.plotly_chart(fig, use_container_width=True)

# Download button for this big dashboard
st.download_button(
    label="📥 Download Main Dashboard as HTML",
    data=fig.to_html(include_plotlyjs='cdn'),
    file_name="telco_main_dashboard.html",
    mime="text/html"
)

st.markdown("---")

# ----------------------------- DETAILED DISTRIBUTION ANALYSIS -----------------------------
st.subheader("2. Detailed Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("**Total Charges by Contract Type (Outliers Removed)**")
    mu, sigma = df_filtered['totalcharges'].mean(
    ), df_filtered['totalcharges'].std()
    clean = df_filtered[(df_filtered['totalcharges'] > mu - 3*sigma) &
                        (df_filtered['totalcharges'] < mu + 3*sigma)]

    fig_box = px.box(clean, x='type', y='totalcharges', color='type',
                     title="Total Charges Distribution per Contract")
    fig_box.update_xaxes(tickangle=30)
    st.plotly_chart(fig_box, use_container_width=True)
    st.download_button("📥 Download Box Plot",
                       fig_box.to_html(), "box_totalcharges.html")

with col2:
    st.write("**Active vs Inactive Customers per Contract Type**")
    count_df = df_filtered.groupby(['type', 'objective'])[
        'customerid'].count().reset_index(name='customers')
    count_df['status'] = count_df['objective'].map(
        {1: 'Active', 0: 'Inactive'})

    fig_bar1 = px.bar(count_df, x='type', y='customers', color='status',
                      barmode='group', text='customers',
                      title="Customer Status by Contract Type")
    fig_bar1.update_traces(textposition='outside')
    st.plotly_chart(fig_bar1, use_container_width=True)
    st.download_button("📥 Download Status Bar",
                       fig_bar1.to_html(), "bar_status.html", key="dl1")

# Std deviation of tenure (time in service)
st.write("**Standard Deviation of Tenure (Days) – Active vs Inactive**")
std_df = df_filtered.groupby(['type', 'objective'])[
    'timeservicedays'].std().reset_index(name='timeservicedays_std')
std_df['status'] = std_df['objective'].map({1: 'Active', 0: 'Inactive'})

fig_bar2 = px.bar(std_df, x='type', y='timeservicedays_std', color='status',
                  barmode='group', title="Variability in Customer timeservicedays by Status")
st.plotly_chart(fig_bar2, use_container_width=True)
st.download_button("📥 Download Tenure Std Bar",
                   fig_bar2.to_html(), "bar_tenure_std.html")


# Prepare data


st.write("**Active vs Churned Customers per Contract Type**")

col1, col2 = st.columns(2)
data = (
    df_data.loc[:, ['beginyear', 'beginmonth', 'customerid', 'objective']]
    .sort_values(['beginyear', 'beginmonth'], ascending=True)
    .groupby(by=['beginyear', 'beginmonth', 'objective'], as_index=False)
    .count()
)


data = data[data['objective'].isin(status_filter) &
            data['beginmonth'].isin(month_filter) &
            data['beginyear'].isin(year_filter)]
data.sort_values(['beginyear', 'beginmonth'], ascending=True)
with col1:
    data['status'] = data['objective'].map({1: 'Active', 0: 'Inactive'})
    fig_sta1 = px.line(data, x='beginyear', y='customerid', color='status',
                       text='customerid',
                       title="Customer Status by Contract Type")
    st.plotly_chart(fig_sta1, use_container_width=True)
    st.download_button("📥 Download year distribution bar",
                       fig_sta1.to_html(), "bar_beginin_year.html")
with col2:
    data['status'] = data['objective'].map({1: 'Active', 0: 'Inactive'})

    fig_sta2 = px.line(data, x='beginmonth', y='customerid', color='status',
                       text='customerid',
                       title="Customer Status by Contract Type")
    st.plotly_chart(fig_sta2, use_container_width=True)
    st.download_button("📥 Download year distribution bar 2",
                       fig_sta2.to_html(), "bar_beginin_year2.html")
# ----------------------------- DATA DOWNLOAD -----------------------------
st.markdown("---")
st.subheader("💾 Download Data")
st.download_button(
    label="Download Filtered Dataset as CSV",
    data=df_filtered.to_csv(index=False),
    file_name="telco_filtered_data.csv",
    mime="text/csv"
)

st.caption(
    "Dashboard created with ❤️ using Streamlit + Plotly • Fully interactive & downloadable")
