import streamlit as st
import pandas as pd
import altair as alt
import phik

DATA_PATH = 'datasets/data_full.csv'
RANDOM_STATE = 42

# configure default settings of the page
st.set_page_config(layout="wide", page_title="Bank Promo Response", page_icon=":bank:")

# initialize variables to filter data based on TARGET values (separately for each EDA tab except correlation)
if 'TARGET_DEMO' not in st.session_state:
    st.session_state['TARGET_DEMO'] = False
if 'TARGET_MONEY' not in st.session_state:
    st.session_state['TARGET_MONEY'] = False
if 'TARGET_CREDIT' not in st.session_state:
    st.session_state['TARGET_CREDIT'] = False


@st.cache_data
def load_data(data_path: str) -> pd.DataFrame:
    """
    Function loads data with cache.

    :param data_path: data path
    :return:
    """

    data = pd.read_csv(DATA_PATH)

    return data


@st.cache_data
def filter_data(df: pd.DataFrame, target_selected: int) -> pd.DataFrame:
    """
    Function filters data based on TARGET feature value with cache.

    :param df: initial data
    :param target_selected: TARGET feature value
    :return: filtered data
    """

    return df[df["TARGET"] == target_selected]


@st.cache_data
def phik_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function computes and returns phik correlation matrix in long format with cache
    :param df: data
    :return: correlation matrix
    """
    data_types = {'TARGET': 'categorical',
                  'AGE': 'interval',
                  'GENDER': 'categorical',
                  'EDUCATION': 'categorical',
                  'MARITAL_STATUS': 'categorical',
                  'CHILD_TOTAL': 'ordinal',
                  'DEPENDANTS': 'ordinal',
                  'SOCSTATUS_WORK_FL': 'categorical',
                  'SOCSTATUS_PENS_FL': 'categorical',
                  'FL_PRESENCE_FL': 'categorical',
                  'OWN_AUTO': 'ordinal',
                  'WORK_TIME': 'interval',
                  'FAMILY_INCOME': 'ordinal',
                  'PERSONAL_INCOME': 'interval',
                  'LOAN_NUM_TOTAL': 'ordinal',
                  'LOAN_NUM_CLOSED': 'ordinal',
                  'CREDIT': 'interval',
                  'TERM': 'interval',
                  'FST_PAYMENT': 'interval'
                  }

    interval_cols = [col for col, v in data_types.items() if v == 'interval']
    corr_matrix = (df[list(data_types.keys())]
                   .phik_matrix(interval_cols=interval_cols, bins=20)
                   .stack()
                   .reset_index()
                   .rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
    corr_matrix['correlation_label'] = corr_matrix['correlation'].map('{:.1f}'.format)

    return corr_matrix


def from_callback(suffix: str) -> None:
    """
    Function switches session state variable with the name TARGET_{suffix}.

    :param suffix: suffix of the session state variable to switch
    """
    st.session_state[f"TARGET_{suffix}"] = not st.session_state[f"TARGET_{suffix}"]


def bar_chart(source: pd.DataFrame, feature: str, color: str, bin: alt.Bin = None,
              x_title: str = None, y_title: str = None) -> None:
    """
    Function draws and displays bar chart.

    :param source: data to display
    :param feature: feature to visualize
    :param color: bars color
    :param bin: binarize data params
    :param x_title: x label title to display
    :param y_title: y label title to display
    """
    chart = alt.Chart(source).transform_joinaggregate(
        total='sum(count)',
    ).transform_calculate(
        percent="datum.count / datum.total"
    ).mark_bar(color=color).encode(
        alt.X(feature, bin=bin, axis=alt.Axis(labelAngle=0, title=x_title)),
        alt.Y('sum(percent):Q', axis=alt.Axis(format='.0%', title=y_title))
    ).properties(height=250)

    st.altair_chart(chart, use_container_width=True)


def pie_chart(source: pd.DataFrame, feature: str) -> None:
    """
    Function draws and displays pie chart.

    :param source: data to display
    :param feature: feature to visualize
    """

    chart = alt.Chart(source).mark_arc(innerRadius=50).encode(
        theta=alt.Theta("percent", stack='normalize'),
        # color=f"{feature}:N"
        color=alt.Color(f"{feature}:N")
    ).properties(height=250).configure_legend(title=None)

    st.altair_chart(chart, use_container_width=True)


# load data
data = load_data(DATA_PATH)

# title and intro text columns
row1_1, row1_2 = st.columns([1, 2])

with row1_1:
    st.title('Предсказание отклика клиента на промо банка: EDA')

with row1_2:
    st.write(
        """
        #
        Проанализируем, отличаются ли клиенты, откликнувшиеся на промо (и если да, то чем), и есть ли корреляция между 
        признаками и целевой переменной.
        
        Перед EDA данные были [предобработаны](
        https://github.com/kirill-rubashevskiy/bank-promo-response-prediction/blob/main/preprocessing.ipynb): отчищены 
        от дубликатов, пропусков и аномальных значений и объединены в одну таблицу. Данные по полученным и погашенным
        кредитам были аггрегированы.
        """
    )

# EDA sections as tabs
tab1, tab2, tab3, tab4 = st.tabs([
    'Демография, образование, семья и социальный статус',
    'Имущество, доходы и работа',
    'Кредитная история',
    'Корреляция'
])

# demographics, education, family and social status tab
with tab1:
    row1_1, row1_2 = st.columns([1, 2])

    with row1_1:
        st.header('Демография, образование, семья и социальный статус')
        # toggle to switch between data filtered by TARGET value
        st.toggle('Отклик на промо', key='target_demo', on_change=from_callback, kwargs={'suffix': 'DEMO'})

    with row1_2:
        st.write("""
            ##
            Среди откликнувшихся на промо: 
            - выше доля женщин (39% против 34%), 20-летних (29% против 21%), студентов; 
            (9% против 5%), работающих (97% против 90%);
            - ниже доля пенсионеров (6% против 15%) и не имеющих иждивенцев (49% против 55%).
            
            На этапе feature engineering нужно попробовать создать признак "неполное среднее или неоконченное высшее 
            образование" и бинаризировать возраст.
            """
        )

    # demographics and education columns
    row2_1, row2_2, row2_3 = st.columns((1.5, 2.5, 3))

    with row2_1:
        st.subheader('Пол')
        source = filter_data(data, st.session_state.TARGET_DEMO).GENDER.value_counts().reset_index().replace({
            1: 'Мужчины',
            0: 'Женщины'
        })

        bar_chart(source, 'GENDER:N', color='#83c9ff', y_title='percent')

    with row2_2:
        st.subheader('Возраст')
        source = filter_data(data, st.session_state.TARGET_DEMO).AGE.value_counts().reset_index()
        bar_chart(source, 'AGE:Q', color='#83c9ff', bin=alt.Bin(maxbins=10))

    with row2_3:
        st.subheader('Образование')
        source = filter_data(data, st.session_state.TARGET_DEMO).EDUCATION.value_counts().reset_index(
        ).rename(columns={'count': 'percent'})
        pie_chart(source, 'EDUCATION')

    # family and social status columns
    row3_1, row3_2, row3_3, row3_4, row3_5 = st.columns([4, 2.5, 2.5, 1, 1])

    with row3_1:
        st.subheader('Семейное положение')
        source = filter_data(data, st.session_state.TARGET_DEMO).MARITAL_STATUS.value_counts().reset_index(
        ).rename(columns={'count': 'percent'})
        pie_chart(source, 'MARITAL_STATUS')

    with row3_2:
        st.subheader('Дети')
        source = filter_data(data, st.session_state.TARGET_DEMO).CHILD_TOTAL.value_counts().reset_index()
        bar_chart(source, 'CHILD_TOTAL:N', color='#83c9ff', y_title='percent')

    with row3_3:
        st.subheader('Иждивенцы')
        source = filter_data(data, st.session_state.TARGET_DEMO).DEPENDANTS.value_counts().reset_index()
        bar_chart(source, 'DEPENDANTS:N', color='#83c9ff')

    with row3_4:
        st.subheader('Работа')
        source = (filter_data(data, st.session_state.TARGET_DEMO).SOCSTATUS_WORK_FL.value_counts()
                  .reset_index()
                  .replace({1: 'да', 0: 'нет'}))
        bar_chart(source, 'SOCSTATUS_WORK_FL:N', color='#83c9ff')

    with row3_5:
        st.subheader('Пенсионер')
        source = (filter_data(data, st.session_state.TARGET_DEMO).SOCSTATUS_PENS_FL.value_counts()
                  .reset_index()
                  .replace({1: 'да', 0: 'нет'}))
        bar_chart(source, 'SOCSTATUS_PENS_FL:N', color='#83c9ff')

# assets, income and job tab
with tab2:
    row1_1, row1_2 = st.columns([1, 2])

    with row1_1:
        st.header('Имущество, доходы и работа')
        # toggle to switch between data filtered by TARGET value
        st.toggle('Отклик на промо', key='target_money', on_change=from_callback, kwargs={'suffix': 'MONEY'})

    with row1_2:
        st.write(
            """
            ##
            Среди откликнувшихся на промо: 
            - выше персональный и семейный доходы (доля дохода от 20 тыс. 50% против 42%), доля работающих в торговле 
            (22% против 17%), рабочих (25% против 22%) и проработавших на последнем месте до 50 месяцев 
            (62% против 51%);
            - ниже доля специалистов (46% против 51%).
            
            На этапе feature engineering нужно попробовать создать признаки "семейный доход от 20 тыс." и 
            бинаризировать период на текущей работе.
            """
        )

    # assets and income columns
    row2_1, row2_2, row2_3, row2_4 = st.columns([1, 1, 3, 3])

    with row2_1:
        st.subheader('Квартира')
        source = filter_data(data, st.session_state.TARGET_MONEY).FL_PRESENCE_FL.value_counts().reset_index().replace({
            1: 'да',
            0: 'нет'
        })
        bar_chart(source, 'FL_PRESENCE_FL:N', color='#fb9a99', y_title='percent')

    with row2_2:
        st.subheader('Машины')
        source = filter_data(data, st.session_state.TARGET_MONEY).OWN_AUTO.value_counts().reset_index()
        bar_chart(source, 'OWN_AUTO:N', color='#fb9a99')

    with row2_3:
        st.subheader('Персональный доход')
        source = filter_data(data, st.session_state.TARGET_MONEY).PERSONAL_INCOME.value_counts().reset_index()
        bar_chart(source, 'PERSONAL_INCOME:Q', color='#fb9a99', bin=alt.Bin(maxbins=30), x_title='рубли')

    with row2_4:
        st.subheader('Семейный доход')
        source = filter_data(data, st.session_state.TARGET_MONEY).FAMILY_INCOME.value_counts().reset_index(
        ).rename(columns={'count': 'percent'})
        pie_chart(source, 'FAMILY_INCOME')

    # job columns
    row3_1, row3_2, row3_3 = st.columns([1.5, 1.5, 1])

    with row3_1:
        st.subheader('Топ-10 отраслей')
        source = filter_data(data, st.session_state.TARGET_MONEY)
        # filter out data re not working clients
        source = source[source.GEN_INDUSTRY != 'not_applicable']
        # aggregate industries not in top10 in a single value 'other industries'
        top10_industries = source.GEN_INDUSTRY.value_counts()[:10].index.tolist()
        source['GEN_INDUSTRY'] = source.GEN_INDUSTRY.apply(lambda x: x if x in top10_industries else 'Другие сферы')
        source = source.GEN_INDUSTRY.value_counts().reset_index().rename(columns={'count': 'percent'})

        pie_chart(source, 'GEN_INDUSTRY')

    with row3_2:
        st.subheader('Топ-10 должностей')
        source = filter_data(data, st.session_state.TARGET_MONEY)
        # filter out data re not working clients
        source = source[source.GEN_TITLE != 'not_applicable']
        # aggregate job titles not in top10 in a single value 'other'
        top10_job_titles = source.GEN_TITLE.value_counts()[:10].index.tolist()
        source['GEN_INDUSTRY'] = source.GEN_TITLE.apply(lambda x: x if x in top10_job_titles else 'Другое')
        source = source.GEN_TITLE.value_counts().reset_index().rename(columns={'count': 'percent'})
        pie_chart(source, 'GEN_TITLE')

    with row3_3:
        st.subheader('На текущей работе')
        source = filter_data(data, st.session_state.TARGET_MONEY).WORK_TIME.value_counts().reset_index()
        # filter out data re not working clients
        source = source[source.WORK_TIME > 0]
        bar_chart(source, 'WORK_TIME:Q', color='#fb9a99', bin=alt.Bin(maxbins=20), x_title='месяцы',
                  y_title='percent')

# credit history tab
with tab3:
    row1_1, row1_2 = st.columns((1, 2))

    with row1_1:
        st.header('Кредитная история')
        # toggle to switch between data filtered by TARGET value
        st.toggle('Отклик на промо', key='target_credit', on_change=from_callback, kwargs={'suffix': 'CREDIT'})

    with row1_2:
        st.write(
            """
            ##
            Среди откликнувшихся на промо выше доля взявших последний кредит на период более 8 месяцев (53% против 45%), 
            взявших один кредит (79% против 74%) и не погасивших не одного кредита (60% против 50%).
            
            На этапе feature engineering нужно попробовать создать соответствующие признаки.
            """
        )

    st.subheader('Последний кредит')

    # last credit columns
    row3_1, row3_2, row3_3 = st.columns([1, 1, 1])

    with row3_1:
        st.subheader('сумма')
        source = filter_data(data, st.session_state.TARGET_CREDIT).CREDIT.value_counts().reset_index()
        bar_chart(source, 'CREDIT:Q', color='#fdbf6f', bin=alt.Bin(maxbins=20),
                  x_title='рубли', y_title='percent')

    with row3_2:
        st.subheader('срок')
        source = filter_data(data, st.session_state.TARGET_CREDIT).TERM.value_counts().reset_index()
        bar_chart(source, 'TERM:Q', color='#fdbf6f', bin=alt.Bin(maxbins=20), x_title='месяцы')

    with row3_3:
        st.subheader('первоначальный взнос')
        source = filter_data(data, st.session_state.TARGET_CREDIT).FST_PAYMENT.value_counts().reset_index()
        bar_chart(source, 'FST_PAYMENT:Q', color='#fdbf6f', bin=alt.Bin(maxbins=20), x_title='рубли')

    # aggregated credits data columns
    row2_1, row2_2, _ = st.columns([1, 1, 1])

    with row2_1:
        st.subheader('Полученные кредиты')
        source = filter_data(data, st.session_state.TARGET_CREDIT).LOAN_NUM_TOTAL.value_counts().reset_index()
        bar_chart(source, 'LOAN_NUM_TOTAL:N', color='#fdbf6f', y_title='percent')

    with row2_2:
        st.subheader('Погашенные кредиты')
        source = filter_data(data, st.session_state.TARGET_CREDIT).LOAN_NUM_CLOSED.value_counts().reset_index()
        bar_chart(source, 'LOAN_NUM_CLOSED:N', color='#fdbf6f')

# correlation tab
with tab4:
    st.header(
        """
        Корреляция
        ##
        """)

    # correlation heatmap and analysis
    row1_1, row1_2, = st.columns([1, 2])

    with row1_1:
        st.write(
            """
            Среди признаков есть числовые, категориальные и порядковые, поэтому будем использовать коэффициент 
            корреляции 𝜙k.
             
            Большинство наблюдаемых высоких корреляций ожидаемы: 
            - между возрастом и социальным статусом; 
            - между полученными и погашенными кредитами; 
            - между параметрами последнего кредита;
            - между возрастом и количеством детей и иждивенцев;
            - между персональным и семейным доходами.
            
            Высокая корреляция между несколькими группами признаков указывает, что стоит произвести их отбор, а также
            применить регуляризацию при обучении линейных моделей.
            
            Низкая корреляция между признаками и целевой переменной указывает, что не стоит ожидать хороших результатов
            от линейных моделей.
            """
        )

    with row1_2:
        source = phik_data(data)
        # correlation heatmap
        plot = alt.Chart(source).mark_rect(strokeOpacity=0).encode(
            x=alt.X('variable:O', axis=alt.Axis(grid=False, title=None, labelLimit=360)),
            y=alt.Y('variable2:O', axis=alt.Axis(grid=False, title=None, labelLimit=360)),
            color=alt.Color('correlation:Q', scale=alt.Scale(scheme='brownbluegreen'))
        ).properties(
            height=760
        )
        text = plot.mark_text(fontSize=16).encode(
            text='correlation_label',
            color=alt.condition(
                ((alt.datum.correlation > 0.75) | (alt.datum.correlation < 0.25)),
                alt.value('white'),
                alt.value('black')
            )
        )
        st.altair_chart(plot + text, use_container_width=True)

