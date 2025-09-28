# 📈 日本株マルチチャート
# 最大12銘柄同時表示 - ドラッグで期間変更可能

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
        
        # VWAPバンドタッチマーカーの追加
        if 'vwap_upper_2' in df.columns and 'vwap_lower_2' in df.columns:
            # バンドタッチ検出
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
            text=f"**📈 日本株マルチチャート - 日足 (ドラッグで期間変更)**",
            font=dict(size=20, color='#2C3E50'),
            x=0.5
        ),
        height=900,
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font=dict(size=10, family="Arial, sans-serif"),
        margin=dict(l=20, r=20, t=60, b=20),
        dragmode='pan',  # ドラッグでパン可能
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
                start_range = max(0, total_length - 20)  # 最新20日分
                x_values = df.index.strftime('%m/%d').tolist()
                
                fig.update_xaxes(
                    type='category',
                    range=[start_range, total_length - 1],  # 最新20日分を表示
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
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #2C3E50; margin-bottom: 5px;'>📈 日本株マルチチャート</h1>
        <p style='color: #7F8C8D; font-size: 18px; margin-top: 0;'>最大12銘柄同時表示 - ドラッグで期間変更可能</p>
    </div>
    """, unsafe_allow_html=True)
    
    # データ読み込み
    stock_data = load_stock_data()
    if stock_data.empty:
        st.error("株式データの読み込みに失敗しました。")
        return
    
    # サイドバー
    with st.sidebar:
        st.header("🔍 銘柄選択")
        
        # ウォッチリスト管理
        st.subheader("📝 ウォッチリスト")
        watchlist_names = get_watchlist_names()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 現在のリストを保存", use_container_width=True):
                if st.session_state.selected_stocks:
                    save_name = st.text_input("保存名を入力:", key="save_name")
                    if save_name and st.button("保存実行", key="save_exec"):
                        save_watchlist(save_name, [stock['ticker'] for stock in st.session_state.selected_stocks])
                        st.success(f"'{save_name}'として保存しました！")
                        st.rerun()
                else:
                    st.warning("選択された銘柄がありません")
        
        with col2:
            if watchlist_names:
                selected_watchlist = st.selectbox("📂 保存済みリスト", ["選択してください"] + watchlist_names, key="load_watchlist")
                if selected_watchlist != "選択してください":
                    if st.button("📂 読み込み", use_container_width=True):
                        loaded_tickers = load_watchlist(selected_watchlist)
                        selected_stocks = []
                        for ticker in loaded_tickers:
                            stock_info = stock_data[stock_data['ticker'] == ticker]
                            if not stock_info.empty:
                                selected_stocks.append({
                                    'ticker': ticker,
                                    'name': stock_info.iloc[0]['name'],
                                    'code': stock_info.iloc[0]['code'],
                                    'sector': stock_info.iloc[0]['sector']
                                })
                        st.session_state.selected_stocks = selected_stocks
                        st.success(f"'{selected_watchlist}'を読み込みました！")
                        st.rerun()
        
        # 検索機能
        st.subheader("🔍 銘柄検索")
        search_method = st.radio("検索方法:", ["銘柄名", "コード", "業種"], horizontal=True)
        
        if search_method == "銘柄名":
            search_term = st.text_input("銘柄名で検索:", placeholder="例: トヨタ")
            if search_term:
                filtered_stocks = stock_data[stock_data['name'].str.contains(search_term, case=False, na=False)]
        elif search_method == "コード":
            search_term = st.text_input("コードで検索:", placeholder="例: 7203")
            if search_term:
                filtered_stocks = stock_data[stock_data['code'].str.contains(search_term, na=False)]
        else:
            sector = st.selectbox("業種を選択:", [""] + sorted(stock_data['sector'].dropna().unique().tolist()))
            if sector:
                filtered_stocks = stock_data[stock_data['sector'] == sector]
            else:
                filtered_stocks = pd.DataFrame()
        
        # 検索結果表示
        if 'search_term' in locals() and search_term:
            if not filtered_stocks.empty:
                st.write(f"検索結果: {len(filtered_stocks)}件")
                for _, stock in filtered_stocks.head(10).iterrows():
                    if st.button(f"➕ {stock['name']} ({stock['code']})", key=f"add_{stock['ticker']}", use_container_width=True):
                        if len(st.session_state.selected_stocks) < 12:
                            if not any(s['ticker'] == stock['ticker'] for s in st.session_state.selected_stocks):
                                st.session_state.selected_stocks.append({
                                    'ticker': stock['ticker'],
                                    'name': stock['name'],
                                    'code': stock['code'],
                                    'sector': stock['sector']
                                })
                                st.success(f"{stock['name']}を追加しました！")
                                st.rerun()
                            else:
                                st.warning("既に選択済みです")
                        else:
                            st.warning("最大12銘柄まで選択可能です")
                if len(filtered_stocks) > 10:
                    st.info("上位10件を表示中")
            else:
                st.info("該当する銘柄が見つかりません")
        elif 'sector' in locals() and sector:
            if not filtered_stocks.empty:
                st.write(f"該当銘柄: {len(filtered_stocks)}件")
                for _, stock in filtered_stocks.head(10).iterrows():
                    if st.button(f"➕ {stock['name']} ({stock['code']})", key=f"add_{stock['ticker']}", use_container_width=True):
                        if len(st.session_state.selected_stocks) < 12:
                            if not any(s['ticker'] == stock['ticker'] for s in st.session_state.selected_stocks):
                                st.session_state.selected_stocks.append({
                                    'ticker': stock['ticker'],
                                    'name': stock['name'],
                                    'code': stock['code'],
                                    'sector': stock['sector']
                                })
                                st.success(f"{stock['name']}を追加しました！")
                                st.rerun()
                            else:
                                st.warning("既に選択済みです")
                        else:
                            st.warning("最大12銘柄まで選択可能です")
                if len(filtered_stocks) > 10:
                    st.info("上位10件を表示中")
        
        # 選択済み銘柄一覧
        st.subheader(f"📋 選択済み銘柄 ({len(st.session_state.selected_stocks)}/12)")
        if st.session_state.selected_stocks:
            for i, stock in enumerate(st.session_state.selected_stocks):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{stock['name']} ({stock['code']})")
                with col2:
                    if st.button("❌", key=f"remove_{i}", help="削除"):
                        st.session_state.selected_stocks.pop(i)
                        st.rerun()
            
            if st.button("🗑️ 全て削除", use_container_width=True):
                st.session_state.selected_stocks = []
                st.rerun()
        else:
            st.info("銘柄が選択されていません")
    
    # メインエリア
    if st.session_state.selected_stocks:
        with st.spinner("📊 チャートデータを取得中..."):
            # 各銘柄の株価データを取得
            selected_stocks_data = []
            progress_bar = st.progress(0)
            
            for i, stock in enumerate(st.session_state.selected_stocks):
                progress_bar.progress((i + 1) / len(st.session_state.selected_stocks))
                stock_price_data = get_stock_data(stock['ticker'])
                selected_stocks_data.append({
                    'ticker': stock['ticker'],
                    'name': stock['name'],
                    'code': stock['code'],
                    'sector': stock['sector'],
                    'data': stock_price_data
                })
                time.sleep(0.1)  # API制限対策
            
            progress_bar.empty()
        
        # チャート作成・表示
        fig = create_multi_chart(selected_stocks_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'resetScale2d']
            })
            
            # 簡易統計情報
            st.subheader("📊 簡易統計")
            stats_cols = st.columns(min(4, len(selected_stocks_data)))
            
            for i, stock in enumerate(selected_stocks_data[:4]):
                with stats_cols[i]:
                    if stock['data'] is not None and not stock['data'].empty:
                        latest = stock['data'].iloc[-1]
                        prev = stock['data'].iloc[-2] if len(stock['data']) > 1 else latest
                        change = latest['Close'] - prev['Close']
                        change_pct = (change / prev['Close']) * 100 if prev['Close'] != 0 else 0
                        
                        color = "🔴" if change < 0 else "🟢" if change > 0 else "⚪"
                        
                        st.metric(
                            label=f"{color} {stock['name'][:8]}",
                            value=f"{latest['Close']:,.0f}円",
                            delta=f"{change:+,.0f}円 ({change_pct:+.2f}%)"
                        )
                    else:
                        st.error(f"❌ {stock['name'][:8]}\nデータ取得失敗")
        else:
            st.error("チャートの作成に失敗しました。")
    else:
        # 使い方説明
        st.markdown("""
        <div style='text-align: center; padding: 40px 20px; background-color: #F8F9FA; border-radius: 10px; margin: 20px 0;'>
            <h3 style='color: #495057; margin-bottom: 20px;'>📈 使い方</h3>
            <div style='text-align: left; max-width: 600px; margin: 0 auto;'>
                <p><strong>1️⃣ 銘柄を選択</strong><br>
                サイドバーから銘柄名・コード・業種で検索し、最大12銘柄まで選択できます。</p>
                
                <p><strong>2️⃣ チャート表示</strong><br>
                選択した銘柄のローソク足チャートとVWAPバンドが表示されます。</p>
                
                <p><strong>3️⃣ 期間変更</strong><br>
                チャート上をドラッグして表示期間を変更できます。</p>
                
                <p><strong>4️⃣ ウォッチリスト</strong><br>
                よく見る銘柄セットを保存・読み込みできます。</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
