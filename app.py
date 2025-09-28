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

        # ===== VWAPバンドタッチマーカー追加 =====
        # バンドタッチ検出（High-Low範囲がバンド値と交差）
        if all(col in df.columns for col in ['vwap_upper_2', 'vwap_lower_2', 'vwap_upper_1', 'vwap_lower_1']):
            # 2σバンドタッチ
            touch_u2 = df[(df['High'] >= df['vwap_upper_2']) & (df['Low'] <= df['vwap_upper_2'])]
            touch_l2 = df[(df['High'] >= df['vwap_lower_2']) & (df['Low'] <= df['vwap_lower_2'])]

            # 1σバンドタッチ
            touch_u1 = df[(df['High'] >= df['vwap_upper_1']) & (df['Low'] <= df['vwap_upper_1'])]
            touch_l1 = df[(df['High'] >= df['vwap_lower_1']) & (df['Low'] <= df['vwap_lower_1'])]

            def add_touch_mark(df_touch, y_col, marker, color):
                if df_touch.empty:
                    return
                touch_x_values = df_touch.index.strftime('%m/%d').tolist()
                fig.add_trace(
                    go.Scatter(
                        x=touch_x_values,
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

            # ±2σタッチマーカー（赤色）
            add_touch_mark(touch_u2, 'vwap_upper_2', 'triangle-up', 'rgba(255,107,107,0.9)')
            add_touch_mark(touch_l2, 'vwap_lower_2', 'triangle-down', 'rgba(255,107,107,0.9)')

            # ±1σタッチマーカー（灰色）
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
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); margin-bottom: 30px; border-radius: 10px;'>
        <h1 style='color: white; margin: 0; font-size: 2.5rem;'>📈 日本株マルチチャート</h1>
        <p style='color: white; margin: 5px 0 0 0; opacity: 0.9;'>最大12銘柄同時表示 - ドラッグで期間変更可能</p>
    </div>
    """, unsafe_allow_html=True)

    # 株式データ読み込み
    stocks_df = load_stock_data()

    if stocks_df.empty:
        st.error("株式データの読み込みに失敗しました。data_j.csvファイルを確認してください。")
        return

    # サイドバー設定
    with st.sidebar:
        st.markdown("### 🔧 設定パネル")

        # ウォッチリスト管理
        st.markdown("#### 📋 ウォッチリスト")

        # ウォッチリスト選択
        watchlist_names = get_watchlist_names()
        if watchlist_names:
            selected_watchlist = st.selectbox(
                "保存済みリストを選択",
                [""] + watchlist_names,
                key="watchlist_selector"
            )

            if selected_watchlist and selected_watchlist != "":
                if st.button("📂 ウォッチリストを読み込み", key="load_watchlist"):
                    loaded_tickers = load_watchlist(selected_watchlist)
                    if loaded_tickers:
                        st.session_state.selected_stocks = loaded_tickers
                        st.success(f"✅ {selected_watchlist} を読み込みました")
                        st.rerun()

        # 新しいウォッチリスト保存
        st.markdown("---")
        new_watchlist_name = st.text_input("💾 新しいリスト名", key="new_watchlist")
        if st.button("💾 現在の選択を保存", key="save_watchlist"):
            if new_watchlist_name and st.session_state.selected_stocks:
                save_watchlist(new_watchlist_name, st.session_state.selected_stocks)
                st.success(f"✅ {new_watchlist_name} を保存しました")
                st.rerun()
            else:
                st.warning("⚠️ リスト名と銘柄を選択してください")

        st.markdown("---")

        # 銘柄検索・選択
        st.markdown("#### 🔍 銘柄検索・選択")

        # セクター絞り込み
        sectors = ['全セクター'] + sorted(stocks_df['sector'].dropna().unique().tolist())
        selected_sector = st.selectbox("🏢 業種で絞り込み", sectors, key="sector_filter")

        # セクターでフィルタリング
        if selected_sector != '全セクター':
            filtered_stocks = stocks_df[stocks_df['sector'] == selected_sector]
        else:
            filtered_stocks = stocks_df

        # 検索
        search_query = st.text_input("🔍 銘柄名・コードで検索", key="search_input")

        if search_query:
            mask = (
                filtered_stocks['name'].str.contains(search_query, na=False) |
                filtered_stocks['code'].str.contains(search_query, na=False)
            )
            filtered_stocks = filtered_stocks[mask]

        # 検索結果表示
        if len(filtered_stocks) > 0:
            st.write(f"📊 検索結果: {len(filtered_stocks)}件")

            # 銘柄選択
            for idx, row in filtered_stocks.head(20).iterrows():
                col1, col2 = st.columns([3, 1])

                with col1:
                    stock_label = f"{row['name']} ({row['code']})"
                    st.write(stock_label[:25])

                with col2:
                    stock_info = {
                        'ticker': row['ticker'],
                        'code': row['code'],
                        'name': row['name'],
                        'sector': row['sector']
                    }

                    if stock_info in st.session_state.selected_stocks:
                        if st.button("❌", key=f"remove_{row['code']}", 
                                   help="リストから削除"):
                            st.session_state.selected_stocks.remove(stock_info)
                            st.rerun()
                    else:
                        if len(st.session_state.selected_stocks) < 12:
                            if st.button("➕", key=f"add_{row['code']}", 
                                       help="リストに追加"):
                                st.session_state.selected_stocks.append(stock_info)
                                st.rerun()
                        else:
                            st.button("🔒", key=f"full_{row['code']}", 
                                    help="最大12銘柄まで", disabled=True)
        else:
            st.write("❌ 検索結果がありません")

        st.markdown("---")

        # 選択済み銘柄リスト
        st.markdown("#### 📈 選択済み銘柄")
        st.write(f"**選択中: {len(st.session_state.selected_stocks)}/12銘柄**")

        if st.session_state.selected_stocks:
            for i, stock in enumerate(st.session_state.selected_stocks):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{i+1}. {stock['name'][:15]}")
                with col2:
                    if st.button("🗑️", key=f"del_{i}", help="削除"):
                        st.session_state.selected_stocks.pop(i)
                        st.rerun()

            if st.button("🗑️ 全てクリア", key="clear_all"):
                st.session_state.selected_stocks = []
                st.rerun()
        else:
            st.write("銘柄が選択されていません")

    # メインチャート表示
    if st.session_state.selected_stocks:
        st.markdown("### 📊 リアルタイムチャート")

        # 進捗表示
        progress_bar = st.progress(0)
        status_text = st.empty()

        # データ取得
        selected_stocks_data = []
        total_stocks = len(st.session_state.selected_stocks)

        for i, stock_info in enumerate(st.session_state.selected_stocks):
            progress = (i + 1) / total_stocks
            progress_bar.progress(progress)
            status_text.text(f"データ取得中... {stock_info['name']} ({i+1}/{total_stocks})")

            # データ取得
            stock_data = get_stock_data(stock_info['ticker'])

            selected_stocks_data.append({
                'ticker': stock_info['ticker'],
                'code': stock_info['code'],
                'name': stock_info['name'],
                'sector': stock_info['sector'],
                'data': stock_data
            })

            # 少し待機（API制限対策）
            time.sleep(0.1)

        # 進捗バー削除
        progress_bar.empty()
        status_text.empty()

        # チャート作成・表示
        chart = create_multi_chart(selected_stocks_data)

        if chart:
            st.plotly_chart(chart, use_container_width=True, key="multi_chart")

            # 統計情報表示
            st.markdown("### 📋 銘柄情報")

            stats_data = []
            for stock_data in selected_stocks_data:
                if stock_data['data'] is not None and not stock_data['data'].empty:
                    df = stock_data['data']
                    latest = df.iloc[-1]

                    # 前日比計算
                    if len(df) > 1:
                        prev_close = df.iloc[-2]['Close']
                        change = latest['Close'] - prev_close
                        change_pct = (change / prev_close) * 100
                    else:
                        change = 0
                        change_pct = 0

                    stats_data.append({
                        '銘柄': f"{stock_data['name'][:10]}({stock_data['code']})",
                        '現在値': f"{latest['Close']:.0f}",
                        '前日比': f"{change:+.0f}",
                        '騰落率': f"{change_pct:+.2f}%",
                        '出来高': f"{latest['Volume']:,.0f}",
                        'セクター': stock_data['sector']
                    })

            if stats_data:
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, use_container_width=True, hide_index=True)

        else:
            st.error("❌ チャートの作成に失敗しました")

    else:
        st.markdown("""
        <div style='text-align: center; padding: 40px; background-color: #f8f9fa; border-radius: 10px; margin: 20px 0;'>
            <h3 style='color: #666; margin-bottom: 20px;'>👈 サイドバーから銘柄を選択してください</h3>
            <p style='color: #888;'>
                • 業種での絞り込みや検索機能を活用<br>
                • 最大12銘柄まで同時表示可能<br>
                • ウォッチリストの保存・読み込み機能
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

