# 📈 日本株マルチチャート

最大12銘柄同時表示 - ドラッグで期間変更可能

import streamlit as st

import pandas as pd

import yfinance as yf

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import numpy as np

import json

import os

from datetime import datetime, timedelta

import time

import math

# ページ設定

st.set_page_config(

page_title="日本株マルチチャート",

page_icon="📈",

layout="wide",

initial_sidebar_state="expanded"

)



# カスタムCSS

st.markdown("""

""", unsafe_allow_html=True)

# セッションステート初期化

if 'selected_stocks' not in st.session_state:

st.session_state.selected_stocks = []

@st.cache_data

def load_stock_data():

"""株式データを読み込む"""

try:

df = pd.read_csv('data_j.csv')

df = df[['コード', '銘柄名', '市場・商品区分', '33業種区分']].copy()

df = df.rename(columns={

'コード': 'code',

'銘柄名': 'name',

'市場・商品区分': 'market',

'33業種区分': 'sector'

})

df = df[df['market'].isin(['プライム（内国株式）', 'スタンダード（内国株式）', 'グロース（内国株式）'])]

df['code'] = df['code'].astype(str).str.zfill(4)

df['ticker'] = df['code'] + '.T'

return df

except Exception as e:

st.error(f"データファイルの読み込みエラー: {e}")

return pd.DataFrame()

def calculate_vwap_bands(df, period=20):

"""TradingView風のVWAPバンド計算（Pine Scriptベース）"""

if len(df) < period:

return df

# Typical Price (hlc3)

typical_price = (df['High'] + df['Low'] + df['Close']) / 3

# Price * Volume

price_volume = typical_price * df['Volume']

# 指定期間の移動平均を使用してVWAP計算

sum_pv = price_volume.rolling(window=period).sum()

sum_vol = df['Volume'].rolling(window=period).sum()

vwap_value = sum_pv / sum_vol

# VWAP基準の偏差計算

deviation = typical_price - vwap_value

squared_dev = deviation ** 2

# 加重標準偏差計算

weighted_squared_dev = squared_dev * df['Volume']

sum_weighted_squared_dev = weighted_squared_dev.rolling(window=period).sum()

variance = sum_weighted_squared_dev / sum_vol

std_dev = np.sqrt(variance)

# VWAPとバンドを計算

df['vwap'] = vwap_value

df['vwap_upper_1'] = vwap_value + std_dev

df['vwap_lower_1'] = vwap_value - std_dev

df['vwap_upper_2'] = vwap_value + 2 * std_dev

df['vwap_lower_2'] = vwap_value - 2 * std_dev

return df

@st.cache_data(ttl=300)

def get_stock_data(ticker, period='3mo', interval='1d'):

"""株価データを取得（90日分）"""

try:

stock = yf.Ticker(ticker)

df = stock.history(period=period, interval=interval)

if df.empty:

return None

df = df.dropna()

df = calculate_vwap_bands(df)

return df

except Exception as e:

st.error(f"株価データの取得エラー ({ticker}): {e}")

return None

def create_multi_chart(selected_stocks_data):

"""12銘柄のマルチチャート作成（トレーディングビュー風ドラッグ対応）"""

if not selected_stocks_data or len(selected_stocks_data) == 0:

return None

# 4列×3行のサブプロット作成

fig = make_subplots(

rows=3, cols=4,

shared_xaxes=False,

vertical_spacing=0.08,

horizontal_spacing=0.05,

subplot_titles=[f"{data['name'][:8]}({data['code']})" for data in selected_stocks_data[:12]]

)

for i, stock_data in enumerate(selected_stocks_data[:12]):

if stock_data['data'] is None or stock_data['data'].empty:

continue

df = stock_data['data']

row = (i // 4) + 1

col = (i % 4) + 1

# 休日を詰めるために日付を文字列に変換

x_values = df.index.strftime('%m/%d').tolist()

# ローソク足チャート

fig.add_trace(

go.Candlestick(

x=x_values,

open=df['Open'],

high=df['High'],

low=df['Low'],

close=df['Close'],

name=stock_data['name'],

decreasing={'line': {'color': '#00D4AA'}, 'fillcolor': '#00D4AA'},

increasing={'line': {'color': '#FF6B6B'}, 'fillcolor': '#FF6B6B'},

showlegend=False

),

row=row, col=col

)

# VWAP

if 'vwap' in df.columns and not df['vwap'].isna().all():

fig.add_trace(

go.Scatter(

x=x_values,

y=df['vwap'],

mode='lines',

name=f'VWAP_{i}',

line=dict(color='#0066FF', width=2),

showlegend=False,

hoverinfo='skip'

),

row=row, col=col

)

# VWAPバンド（2σ - 外側、赤色）

if 'vwap_upper_2' in df.columns and not df['vwap_upper_2'].isna().all():

fig.add_trace(

go.Scatter(

x=x_values,

y=df['vwap_upper_2'],

mode='lines',

line=dict(color='rgba(255, 107, 107, 0.8)', width=1, dash='dot'),

showlegend=False,

hoverinfo='skip'

),

row=row, col=col

)

fig.add_trace(

go.Scatter(

x=x_values,

y=df['vwap_lower_2'],

mode='lines',

line=dict(color='rgba(255, 107, 107, 0.8)', width=1, dash='dot'),

fill='tonexty',

fillcolor='rgba(255, 107, 107, 0.1)',

showlegend=False,

hoverinfo='skip'

),

row=row, col=col

)

# VWAPバンド（1σ - 内側、グレー）

if 'vwap_upper_1' in df.columns and not df['vwap_upper_1'].isna().all():

fig.add_trace(

go.Scatter(

x=x_values,

y=df['vwap_upper_1'],

mode='lines',

line=dict(color='rgba(128, 128, 128, 0.6)', width=1, dash='dash'),

showlegend=False,

hoverinfo='skip'

),

row=row, col=col

)

fig.add_trace(

go.Scatter(

x=x_values,

y=df['vwap_lower_1'],

mode='lines',

line=dict(color='rgba(128, 128, 128, 0.6)', width=1, dash='dash'),

fill='tonexty',

fillcolor='rgba(128, 128, 128, 0.1)',

showlegend=False,

hoverinfo='skip'

),

row=row, col=col

)

# ─── バンドタッチ検出とマーカー追加 ───
touch_u2 = df[(df['High'] >= df['vwap_upper_2']) & (df['Low'] <= df['vwap_upper_2'])]
touch_l2 = df[(df['High'] >= df['vwap_lower_2']) & (df['Low'] <= df['vwap_lower_2'])]
touch_u1 = df[(df['High'] >= df['vwap_upper_1']) & (df['Low'] <= df['vwap_upper_1'])]
touch_l1 = df[(df['High'] >= df['vwap_lower_1']) & (df['Low'] <= df['vwap_lower_1'])]

def add_touch_mark(df_touch, y_col, marker, color):
    if df_touch.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=df_touch.index.strftime('%m/%d'),
            y=df_touch[y_col],
            mode='markers',
            marker_symbol=marker,
            marker_size=8,
            marker_color=color,
            name='touch',
            showlegend=False,
            hoverinfo='skip'
        ),
        row=row, col=col
    )

# ±2σタッチ（赤）
add_touch_mark(touch_u2, 'vwap_upper_2', 'triangle-up', 'rgba(255,107,107,0.9)')
add_touch_mark(touch_l2, 'vwap_lower_2', 'triangle-down', 'rgba(255,107,107,0.9)')

# ±1σタッチ（灰）
add_touch_mark(touch_u1, 'vwap_upper_1', 'triangle-up', 'rgba(128,128,128,0.9)')
add_touch_mark(touch_l1, 'vwap_lower_1', 'triangle-down', 'rgba(128,128,128,0.9)')

# レイアウト更新（トレーディングビュー風）

fig.update_layout(

title=dict(

text=f"

**📈 日本株マルチチャート - 日足 (ドラッグで期間変更)**",

font=dict(size=20, color='#2C3E50'),

x=0.5

),

height=900,

template="plotly_white",

paper_bgcolor='rgba(0,0,0,0)',

plot_bgcolor='white',

font=dict(size=10, family="Arial, sans-serif"),

margin=dict(l=20, r=20, t=60, b=20),

dragmode='pan', # ドラッグでパン可能

showlegend=False

)

# 各サブプロットのX軸設定（トレーディングビュー風）

for i in range(1, 13):

row = ((i-1) // 4) + 1

col = ((i-1) % 4) + 1

# 最新20日分を初期表示に設定

if selected_stocks_data and len(selected_stocks_data) > i-1 and selected_stocks_data[i-1]['data'] is not None:

df = selected_stocks_data[i-1]['data']

if not df.empty:

total_length = len(df)

start_range = max(0, total_length - 20) # 最新20日分

x_values = df.index.strftime('%m/%d').tolist()

fig.update_xaxes(

type='category',

range=[start_range, total_length - 1], # 最新20日分を表示

showgrid=True,

gridwidth=0.3,

gridcolor='rgba(128,128,128,0.2)',

tickangle=45,

tickfont=dict(size=8),

rangeslider_visible=False,

row=row, col=col

)

# Y軸の設定

fig.update_yaxes(

showgrid=True,

gridwidth=0.3,

gridcolor='rgba(128,128,128,0.2)',

tickfont=dict(size=8),

row=row, col=col

)

return fig

def save_watchlist(name, tickers):

"""ウォッチリストを保存"""

if not os.path.exists('watchlists'):

os.makedirs('watchlists')

with open(f'watchlists/{name}.json', 'w', encoding='utf-8') as f:

json.dump(tickers, f, ensure_ascii=False, indent=2)

def load_watchlist(name):

"""ウォッチリストを読み込み"""

try:

with open(f'watchlists/{name}.json', 'r', encoding='utf-8') as f:

return json.load(f)

except:

return []

def get_watchlist_names():

"""保存されたウォッチリスト名を取得"""

if not os.path.exists('watchlists'):

return []

files = [f[:-5] for f in os.listdir('watchlists') if f.endswith('.json')]

return files

def main():

# ヘッダー

st.markdown("""

# 📈 日本株マルチチャート

最大12銘柄同時表示 - ドラッグで期間変更可能

---

""")

# データ読み込み

stock_df = load_stock_data()

if stock_df.empty:

st.error("株式データが読み込めませんでした。")

return

# サイドバー

with st.sidebar:

st.header("🔍 銘柄選択")

# 検索機能

search_term = st.text_input("銘柄名/コード検索", placeholder="例: トヨタ、7203")

if search_term:

filtered_df = stock_df[

(stock_df['name'].str.contains(search_term, case=False, na=False)) |

(stock_df['code'].str.contains(search_term, na=False))

]

else:

filtered_df = stock_df

# 業種フィルター

sectors = ['全て'] + sorted(stock_df['sector'].unique().tolist())

selected_sector = st.selectbox("業種フィルター", sectors)

if selected_sector != '全て':

filtered_df = filtered_df[filtered_df['sector'] == selected_sector]

# 市場フィルター

markets = ['全て'] + sorted(stock_df['market'].unique().tolist())

selected_market = st.selectbox("市場フィルター", markets)

if selected_market != '全て':

filtered_df = filtered_df[filtered_df['market'] == selected_market]

# 銘柄リスト表示（最大500件に制限）

display_df = filtered_df.head(500)

st.write(f"表示件数: {len(display_df)} / {len(filtered_df)} 件")

if len(filtered_df) > 500:

st.warning("⚠️ 検索結果が多すぎます。上位500件のみ表示しています。")

# 銘柄選択

for _, row in display_df.iterrows():

display_name = f"{row['name']} ({row['code']})"

if st.checkbox(display_name, key=row['code']):

if row['code'] not in [s['code'] for s in st.session_state.selected_stocks]:

st.session_state.selected_stocks.append({

'code': row['code'],

'name': row['name'],

'ticker': row['ticker']

})

else:

st.session_state.selected_stocks = [

s for s in st.session_state.selected_stocks if s['code'] != row['code']

]

st.rerun()

# 現在の選択銘柄表示

st.subheader(f"📊 選択中の銘柄 ({len(st.session_state.selected_stocks)}/12)")

if st.session_state.selected_stocks:

for stock in st.session_state.selected_stocks:

if st.button(f"❌ {stock['name']} ({stock['code']})", key=f"remove_{stock['code']}"):

st.session_state.selected_stocks = [

s for s in st.session_state.selected_stocks if s['code'] != stock['code']

]

st.rerun()

# 全削除ボタン

if st.button("🗑️ 全て削除"):

st.session_state.selected_stocks = []

st.rerun()

# ウォッチリスト機能

st.subheader("📝 ウォッチリスト")

# 保存

if st.session_state.selected_stocks:

watchlist_name = st.text_input("ウォッチリスト名", placeholder="例: 高配当株")

if st.button("💾 保存"):

if watchlist_name:

tickers = [s['ticker'] for s in st.session_state.selected_stocks]

save_watchlist(watchlist_name, tickers)

st.success(f"ウォッチリスト '{watchlist_name}' を保存しました！")

time.sleep(1)

st.rerun()

else:

st.error("ウォッチリスト名を入力してください")

# 読み込み

watchlist_names = get_watchlist_names()

if watchlist_names:

selected_watchlist = st.selectbox("保存済みウォッチリスト", ['選択してください'] + watchlist_names)

if selected_watchlist != '選択してください':

if st.button("📂 読み込み"):

tickers = load_watchlist(selected_watchlist)

st.session_state.selected_stocks = []

for ticker in tickers:

code = ticker.replace('.T', '')

stock_info = stock_df[stock_df['code'] == code]

if not stock_info.empty:

st.session_state.selected_stocks.append({

'code': code,

'name': stock_info.iloc[0]['name'],

'ticker': ticker

})

st.success(f"ウォッチリスト '{selected_watchlist}' を読み込みました！")

time.sleep(1)

st.rerun()

# メインコンテンツ

if not st.session_state.selected_stocks:

st.info("📌 左のサイドバーから銘柄を選択してください（最大12銘柄）")

return

if len(st.session_state.selected_stocks) > 12:

st.warning("⚠️ 最大12銘柄までです。超過分は表示されません。")

# 株価データ取得

st.info("📊 株価データを取得中...")

selected_stocks_data = []

for stock in st.session_state.selected_stocks[:12]:

with st.spinner(f"{stock['name']} のデータを取得中..."):

data = get_stock_data(stock['ticker'])

selected_stocks_data.append({

'code': stock['code'],

'name': stock['name'],

'ticker': stock['ticker'],

'data': data

})

# チャート作成・表示

chart = create_multi_chart(selected_stocks_data)

if chart:

st.plotly_chart(chart, use_container_width=True, config={

'displayModeBar': True,

'displaylogo': False,

'modeBarButtonsToRemove': ['select2d', 'lasso2d'],

'toImageButtonOptions': {

'format': 'png',

'filename': 'stock_chart',

'height': 900,

'width': 1600,

'scale': 2

}

})

# データテーブル表示オプション

if st.checkbox("📋 データテーブルを表示"):

for stock_data in selected_stocks_data:

if stock_data['data'] is not None and not stock_data['data'].empty:

st.subheader(f"{stock_data['name']} ({stock_data['code']})")

df_display = stock_data['data'].copy()

df_display = df_display.round(2)

df_display.index = df_display.index.strftime('%Y-%m-%d')

st.dataframe(df_display.tail(10), use_container_width=True)

else:

st.error("チャートの作成に失敗しました。")

if __name__ == "__main__":

main()

