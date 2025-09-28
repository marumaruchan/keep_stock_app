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
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 2rem;
    color: white;
}
.selected-stock {
    background-color: #e8f4fd;
    padding: 0.5rem;
    margin: 0.25rem 0;
    border-radius: 5px;
    border-left: 4px solid #1f77b4;
}
.search-result {
    background-color: #f8f9fa;
    padding: 0.25rem;
    margin: 0.1rem 0;
    border-radius: 3px;
}
</style>
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
        # ─── VWAPバンドタッチ検出 ───
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
            text=f"<b>📈 日本株マルチチャート - 日足 (ドラッグで期間変更)</b>",
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
        tickfont=dict(size=8)
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
    <div class="main-header">
        <h1>📈 日本株マルチチャート</h1>
        <p>最大12銘柄同時表示 - ドラッグで期間変更可能</p>
    </div>
    """, unsafe_allow_html=True)
    
    # データ読み込み
    stock_df = load_stock_data()
    
    if stock_df.empty:
        st.error("株式データの読み込みに失敗しました。")
        return
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # 選択済み銘柄表示
        st.subheader("📋 選択中の銘柄")
        if st.session_state.selected_stocks:
            for i, ticker in enumerate(st.session_state.selected_stocks):
                stock_info = stock_df[stock_df['ticker'] == ticker]
                if not stock_info.empty:
                    name = stock_info.iloc[0]['name']
                    code = stock_info.iloc[0]['code']
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f'<div class="selected-stock">{code} {name[:12]}</div>', 
                                  unsafe_allow_html=True)
                    with col2:
                        if st.button("❌", key=f"remove_{i}"):
                            st.session_state.selected_stocks.remove(ticker)
                            st.rerun()
        else:
            st.info("銘柄を選択してください")
        
        if st.button("🗑️ 全て削除"):
            st.session_state.selected_stocks = []
            st.rerun()
        
        # 銘柄検索エリア
        st.subheader("🔍 銘柄検索・追加")
        search_term = st.text_input("銘柄検索", placeholder="銘柄名またはコードを入力")
        
        # 検索結果表示
        if search_term:
            filtered_df = stock_df[
                (stock_df['name'].str.contains(search_term, na=False, case=False)) |
                (stock_df['code'].str.contains(search_term, na=False, case=False))
            ].head(20)
            
            st.write("**検索結果:**")
            for _, row in filtered_df.iterrows():
                if len(st.session_state.selected_stocks) >= 12:
                    st.warning("最大12銘柄まで選択可能です")
                    break
                
                if row['ticker'] not in st.session_state.selected_stocks:
                    if st.button(f"➕ {row['code']} {row['name'][:20]}", key=f"add_{row['ticker']}"):
                        st.session_state.selected_stocks.append(row['ticker'])
                        st.rerun()
                else:
                    st.write(f"✅ {row['code']} {row['name'][:20]} (選択済み)")
        
        # ウォッチリスト管理
        st.subheader("⭐ ウォッチリスト")
        
        # 既存のウォッチリスト
        watchlist_names = get_watchlist_names()
        if watchlist_names:
            selected_watchlist = st.selectbox(
                "ウォッチリスト選択",
                [""] + watchlist_names
            )
            
            if selected_watchlist:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 読み込み"):
                        watchlist_tickers = load_watchlist(selected_watchlist)
                        st.session_state.selected_stocks = watchlist_tickers[:12]
                        st.success(f"'{selected_watchlist}'を読み込みました")
                        st.rerun()
                
                with col2:
                    if st.button("💾 上書き保存"):
                        save_watchlist(selected_watchlist, st.session_state.selected_stocks)
                        st.success(f"'{selected_watchlist}'を更新しました")
        
        # 新規ウォッチリスト作成
        with st.expander("新しいリスト作成"):
            new_watchlist_name = st.text_input("新しいリスト名")
            if st.button("💾 現在の選択で作成"):
                if new_watchlist_name and st.session_state.selected_stocks:
                    save_watchlist(new_watchlist_name, st.session_state.selected_stocks)
                    st.success(f"'{new_watchlist_name}'を作成しました")
                    st.rerun()
                else:
                    st.error("リスト名と銘柄選択が必要です")
    
    # メインエリア
    if st.session_state.selected_stocks:
        st.subheader("📊 マルチチャート - 日足（90日間データ）")
        
        # 操作ガイド
        st.info("💡 **操作方法:** チャートをドラッグして期間移動、マウスホイールで拡大縮小、ダブルクリックでズームリセット")
        
        with st.spinner("チャートを読み込み中..."):
            # 各銘柄のデータを取得
            selected_stocks_data = []
            progress_bar = st.progress(0)
            
            for i, ticker in enumerate(st.session_state.selected_stocks):
                stock_info = stock_df[stock_df['ticker'] == ticker]
                if not stock_info.empty:
                    name = stock_info.iloc[0]['name']
                    code = stock_info.iloc[0]['code']
                else:
                    name = ticker
                    code = ticker.replace('.T', '')
                
                stock_data = get_stock_data(ticker, '3mo', '1d')  # 90日分取得
                
                selected_stocks_data.append({
                    'ticker': ticker,
                    'name': name,
                    'code': code,
                    'data': stock_data
                })
                
                progress_bar.progress((i + 1) / len(st.session_state.selected_stocks))
            
            progress_bar.empty()
            
            # マルチチャート作成
            multi_chart = create_multi_chart(selected_stocks_data)
            
            if multi_chart:
                st.plotly_chart(multi_chart, use_container_width=True)
                
                # 銘柄別最新価格
                st.subheader("💰 銘柄別最新価格")
                
                cols = st.columns(4)
                for i, stock_data in enumerate(selected_stocks_data[:12]):
                    with cols[i % 4]:
                        if stock_data['data'] is not None and not stock_data['data'].empty:
                            latest = stock_data['data'].iloc[-1]
                            prev_close = stock_data['data'].iloc[-2]['Close'] if len(stock_data['data']) > 1 else latest['Close']
                            change = latest['Close'] - prev_close
                            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                            
                            st.metric(
                                label=f"{stock_data['code']} {stock_data['name'][:8]}",
                                value=f"¥{latest['Close']:,.0f}",
                                delta=f"{change_pct:+.2f}%"
                            )
                        else:
                            st.metric(
                                label=f"{stock_data['code']} {stock_data['name'][:8]}",
                                value="データなし",
                                delta=None
                            )
            else:
                st.error("チャートの作成に失敗しました")
    else:
        st.info("左側のサイドバーから銘柄を選択してください（最大12銘柄）")
    
    # フッター
    st.markdown("---")
    st.markdown("""
    🎯 **操作方法:** 
    - **ドラッグ**: チャートをドラッグして期間を移動
    - **ズーム**: マウスホイールで拡大縮小
    - **リセット**: ダブルクリックでズームリセット
    - **データ範囲**: 90日分のデータを格納、初期表示は最新20日分
    """)

if __name__ == "__main__":
    main()



