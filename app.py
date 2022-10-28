##
import dash
import pandas as pd
import investpy as inv
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gc

from dash import dcc, Output, Input
from dash import html
from dateutil.relativedelta import relativedelta
from datetime import datetime
##
def get_log_return(ticker, column_name):
    np.random.seed(0)
    return_col = 'log_ret_' + ticker
    df = yf.download(ticker, start=(datetime.now() - relativedelta(years=5)).strftime("%Y-%m-%d"), end=datetime.now().strftime("%Y-%m-%d"))
    df['Date'] = pd.to_datetime(df.index)
    df['log_ret'] = np.log(df[column_name]) - np.log(df[column_name].shift(1))
    df['Color'] = np.where(df["log_ret"]<0, 'Loss', 'Gain')
    df['Volatility'] =  df['log_ret'].rolling(window=252).std()*np.sqrt(252)*100
    df[return_col] = df['log_ret']
    return df

def get_model_columns(ticker,df):
    df = df[['log_ret_' + ticker,'Date']]
    return df

## Nifty 50 Data
df_NIFTY50 = get_log_return("^NSEI",'Close')
df_NIFTY50_mod = get_model_columns("^NSEI",df_NIFTY50)
gc.collect()

## BIST 100 Data
df_bist = get_log_return("XU100.IS",'Close')
df_bist_mod = get_model_columns("XU100.IS",df_bist)
gc.collect()

## Dow
df_dji = get_log_return("^DJI",'Close')
df_dji_mod = get_model_columns("^DJI",df_dji)
gc.collect()

## NYSE
df_nya = get_log_return("^NYA",'Close')
df_nya_mod = get_model_columns("^NYA",df_nya)
gc.collect()

## FTSE
df_ftse = get_log_return("^FTSE",'Close')

## German DAX
df_dax = get_log_return("^GDAXI",'Close')
df_dax_mod = get_model_columns("^GDAXI",df_dax)
gc.collect()

## KOSPI
df_kospi = get_log_return("^KS11",'Close')
df_kospi_mod = get_model_columns("^KS11",df_kospi)
gc.collect()

## Dollar Index
df_usd = get_log_return("DX-Y.NYB",'Close')
df_usd_mod = get_model_columns("DX-Y.NYB",df_usd)
gc.collect()

## United States Brent Oil Fund
df_oil = get_log_return("BNO",'Close')
df_oil_mod = get_model_columns("BNO",df_oil)
gc.collect()

## Nikkei
df_nikkei = get_log_return("^N225",'Close')
df_nikkei_mod = get_model_columns("^N225",df_nikkei)
gc.collect()

## Gold ETF
df_au = get_log_return("AAAU",'Close')
df_au_mod = get_model_columns("AAAU",df_au)
gc.collect()

## SILVERBEES.NS
df_ag = get_log_return("SILVERBEES.NS",'Close')
df_ag_mod = get_model_columns("SILVERBEES.NS",df_ag)
gc.collect()

## Bursa Malaysia ^KLSE
df_my = get_log_return("^KLSE",'Close')
df_my_mod = get_model_columns("^KLSE",df_my)
gc.collect()

##
external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "Nifty50 Analytics understanding the maverick market!"
app.layout = html.Div(
    children=[
        html.H1(children="Nifty 50 Analytics", style={"fontSize":"32px", "color":"blue", "text-align":"center"}),
        html.P(
            children="Analyze Nifty 50 price Trend with Volume", className="header-description"
        ),
        html.Div(
            children=[
                html.Div(
                    children="Date Range",
                ),
                dcc.DatePickerRange(
                    id="date-range",
                    min_date_allowed=df_NIFTY50.Date.min().date(),
                    max_date_allowed=df_NIFTY50.Date.max().date(),
                    start_date=df_NIFTY50.Date.min().date(),
                    end_date=df_NIFTY50.Date.max().date(),
                )
            ], className="menu"
        ),
        dcc.Graph(
            id="price-chart"
        ),
        dcc.Graph(
            id="volume-chart"
        ),
        dcc.Graph(
            id="vix-chart"
        ),
        dcc.Graph(
            id="ret-chart"
        ),
        dcc.Graph(
            id="ret-bist"
        ),
        dcc.Graph(
            id="ret-dji"
        ),
        dcc.Graph(
            id="ret-nya"
        ),
        dcc.Graph(
            id="ret-ftse"
        ),
        dcc.Graph(
            id="ret-dax"
        ),
        dcc.Graph(
            id="ret-kospi"
        ),
        dcc.Graph(
            id="ret-usd"
        ),
        dcc.Graph(
            id="ret-oil"
        ),
        dcc.Graph(
            id="ret-nikkei"
        ),
        dcc.Graph(
            id="ret-au"
        ),
        dcc.Graph(
            id="ret-ag"
        ),
        dcc.Graph(
            id="ret-my"
        ),
    ]
)
@app.callback(
    [Output("price-chart", "figure"), Output("volume-chart", "figure"), Output("vix-chart", "figure"),Output("ret-chart", "figure"),Output("ret-bist", "figure"),Output("ret-dji", "figure"),Output("ret-nya", "figure"),Output("ret-ftse", "figure"),Output("ret-dax", "figure"),Output("ret-kospi", "figure"),Output("ret-usd", "figure"),Output("ret-oil", "figure"),Output("ret-nikkei", "figure"),Output("ret-au", "figure"),Output("ret-ag", "figure"),Output("ret-my", "figure")],
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)


def update_charts(start_date, end_date):
    def get_filtered_data(start_date, end_date,df):
        mask = (
            (df.Date >= start_date)
            & (df.Date <= end_date))
        return df.loc[mask, :]
        
    filtered_data = get_filtered_data(start_date, end_date, df=df_NIFTY50)
    filtered_data_bist = get_filtered_data(start_date, end_date, df=df_bist)
    filtered_data_dji = get_filtered_data(start_date, end_date, df=df_dji)
    filtered_data_nya = get_filtered_data(start_date, end_date, df=df_nya)
    filtered_data_ftse = get_filtered_data(start_date, end_date, df=df_ftse)
    filtered_data_dax = get_filtered_data(start_date, end_date, df=df_dax)
    filtered_data_kospi = get_filtered_data(start_date, end_date, df=df_kospi)
    filtered_data_usd = get_filtered_data(start_date, end_date, df=df_usd)
    filtered_data_oil = get_filtered_data(start_date, end_date, df=df_oil)
    filtered_data_nikkei = get_filtered_data(start_date, end_date, df=df_nikkei)
    filtered_data_au = get_filtered_data(start_date, end_date, df=df_au)
    filtered_data_ag = get_filtered_data(start_date, end_date, df=df_ag)
    filtered_data_my = get_filtered_data(start_date, end_date, df=df_my)

    price_chart_figure = {
        "data": [
            {
                "x": filtered_data["Date"],
                "y": filtered_data["Close"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {
                "text": "Close of Nifty50",
            },
            "colorway": ["#17B897"],
        },
    }

    volume_chart_figure = {
        "data": [
            {
                "x": filtered_data["Date"],
                "y": filtered_data["Volume"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {
                "text": "Volume Nifty50",
            },
            "colorway": ["#E12D39"],
        },
    }

    scatter = px.scatter(
        filtered_data,
        x="Date",
        y="log_ret",
        color="Color",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Daily Return for Nifty 50",
    )

    scatter_bist100 = px.scatter(
        filtered_data_bist,
        x="Date",
        y="log_ret",
        color="Color",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Daily Return for BIST 100",
    )

    scatter_dji = px.scatter(
        filtered_data_dji,
        x="Date",
        y="log_ret",
        color="Color",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Daily Return for DJI",
    )

    scatter_nya = px.scatter(
        filtered_data_nya,
        x="Date",
        y="log_ret",
        color="Color",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Daily Return for NYA",
    )

    scatter_ftse = px.scatter(
        filtered_data_ftse,
        x="Date",
        y="log_ret",
        color="Color",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Daily Return for FTSE",
    )

    scatter_dax = px.scatter(
        filtered_data_dax,
        x="Date",
        y="log_ret",
        color="Color",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Daily Return for DAX",
    )

    scatter_kospi = px.scatter(
        filtered_data_kospi,
        x="Date",
        y="log_ret",
        color="Color",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Daily Return for KOSPI",
    )

    scatter_usd = px.scatter(
        filtered_data_usd,
        x="Date",
        y="log_ret",
        color="Color",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Daily Return for Dollar Index",
    )

    scatter_oil = px.scatter(
        filtered_data_oil,
        x="Date",
        y="log_ret",
        color="Color",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Daily Return for Brent Oil",
    )

    scatter_nikkei = px.scatter(
        filtered_data_nikkei,
        x="Date",
        y="log_ret",
        color="Color",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Daily Return for Nikkei",
    )

    scatter_au = px.scatter(
        filtered_data_au,
        x="Date",
        y="log_ret",
        color="Color",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Daily Return for Gold",
    )

    scatter_ag = px.scatter(
        filtered_data_ag,
        x="Date",
        y="log_ret",
        color="Color",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Daily Return for Silver",
    )

    scatter_my = px.scatter(
        filtered_data_my,
        x="Date",
        y="log_ret",
        color="Color",
        color_continuous_scale=px.colors.sequential.Plotly3,
        title="Daily Return for Malaysian Bursa",
    )

    return price_chart_figure, volume_chart_figure,vix_chart_figure,scatter,scatter_bist100,scatter_dji,scatter_nya,scatter_ftse,scatter_dax,scatter_kospi,scatter_usd,scatter_oil,scatter_nikkei,scatter_au,scatter_ag, scatter_my


if __name__ == "__main__":
    app.run_server(debug=True)

