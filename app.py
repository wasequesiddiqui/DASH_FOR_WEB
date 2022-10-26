##
import dash
import pandas as pd
import investpy as inv
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, Output, Input
from dash import html

from dateutil.relativedelta import relativedelta

from datetime import datetime
##
def get_log_return(ticker, column_name):
    np.random.seed(0)
    df = yf.download(ticker, start=(datetime.now() - relativedelta(years=5)).strftime("%Y-%m-%d"), end=datetime.now().strftime("%Y-%m-%d"))
    df['Date'] = pd.to_datetime(df.index)
    df['log_ret'] = np.log(df[column_name]) - np.log(df[column_name].shift(1))
    df['Color'] = np.where(df["log_ret"]<0, 'Loss', 'Gain')
    return df

## Nifty 50 Data
df_NIFTY50 = get_log_return("^NSEI",'Close')

## BIST 100 Data
df_bist = get_log_return("XU100.IS",'Close')

## Dow
df_dji = get_log_return("^DJI",'Close')

## NYSE
df_nya = get_log_return("^NYA",'Close')

## FTSE

## German DAX

## KOSPI

## Dollar Index

## United States Brent Oil Fund

## Nikkei

## Gold ETF

## SILVERBEES.NS

## Bursa Malaysia ^KLSE

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
    ]
)
@app.callback(
    [Output("price-chart", "figure"), Output("volume-chart", "figure"),Output("ret-chart", "figure"),Output("ret-bist", "figure"),Output("ret-dji", "figure"),Output("ret-nya", "figure")],
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

    return price_chart_figure, volume_chart_figure,scatter,scatter_bist100,scatter_dji,scatter_nya


if __name__ == "__main__":
    app.run_server(debug=True)

