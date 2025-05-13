import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def plot_candlestick_with_indicators(df, ticker, start_date, end_date):
    """Create an enhanced candlestick chart with volume and technical indicators"""
    # Filter data based on selected date range
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    filtered_df = df.loc[mask]
    
    if filtered_df.empty:
        raise ValueError("No data available for the selected date range")
    
    # Create subplots with 3 rows (price, volume, indicators)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # 1. Candlestick Chart (Top Panel)
    fig.add_trace(go.Candlestick(
        x=filtered_df.index,
        open=filtered_df['Open'],
        high=filtered_df['High'],
        low=filtered_df['Low'],
        close=filtered_df['Close'],
        name='Price',
        increasing_line_color='#2ECC71',
        decreasing_line_color='#E74C3C'
    ), row=1, col=1, secondary_y=False)
    
    # Add moving averages if available
    if 'SMA_20' in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_df['SMA_20'],
            mode='lines',
            name='20-SMA',
            line=dict(color='orange', width=1.5),
            opacity=0.8
        ), row=1, col=1, secondary_y=False)
        
    if 'EMA_50' in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_df['EMA_50'],
            mode='lines',
            name='50-EMA',
            line=dict(color='purple', width=1.5),
            opacity=0.8
        ), row=1, col=1, secondary_y=False)
    
    # Add Bollinger Bands if available
    if all(col in filtered_df.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_df['BB_upper'],
            mode='lines',
            line=dict(width=1, color='gray'),
            name='BB Upper',
            opacity=0.7
        ), row=1, col=1, secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_df['BB_lower'],
            mode='lines',
            line=dict(width=1, color='gray'),
            name='BB Lower',
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.2)',
            opacity=0.7
        ), row=1, col=1, secondary_y=False)
    
    # 2. Volume Bars (Middle Panel)
    if 'Volume' in filtered_df.columns:
        # Color volume bars based on price movement
        colors = np.where(filtered_df['Close'] > filtered_df['Open'], '#2ECC71', '#E74C3C')
        fig.add_trace(go.Bar(
            x=filtered_df.index,
            y=filtered_df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.6
        ), row=2, col=1)
    
    # 3. Technical Indicators (Bottom Panel)
    if 'RSI' in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='#3498DB', width=1.5)
        ), row=3, col=1)
        
        # Add RSI thresholds
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Add trading signals
    if 'MACD' in filtered_df.columns and 'Signal_Line' in filtered_df.columns:
        # Highlight bullish crosses (MACD crosses above Signal)
        bullish = filtered_df[(filtered_df['MACD'] > filtered_df['Signal_Line']) & 
                            (filtered_df['MACD'].shift(1) <= filtered_df['Signal_Line'].shift(1))]
        fig.add_trace(go.Scatter(
            x=bullish.index,
            y=bullish['Low'] * 0.98,
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='green',
                line=dict(width=1, color='DarkGreen')
            ),
            name='Bullish Signal',
            opacity=0.8
        ), row=1, col=1, secondary_y=False)
        
        # Highlight bearish crosses (MACD crosses below Signal)
        bearish = filtered_df[(filtered_df['MACD'] < filtered_df['Signal_Line']) & 
                            (filtered_df['MACD'].shift(1) >= filtered_df['Signal_Line'].shift(1))]
        fig.add_trace(go.Scatter(
            x=bearish.index,
            y=bearish['High'] * 1.02,
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color='red',
                line=dict(width=1, color='DarkRed')
            ),
            name='Bearish Signal',
            opacity=0.8
        ), row=1, col=1, secondary_y=False)
    
    # Update layout
    fig.update_layout(
        title=f'📈 {ticker} - {start_date.strftime("%d %b %Y")} to {end_date.strftime("%d %b %Y")}',
        template='plotly_dark',
        height=800,
        showlegend=True,
        hovermode='x unified',
        xaxis_range=[start_date, end_date],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, b=50, t=80),
        xaxis_rangeslider_visible=False
    )
    
    # Update y-axes titles
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    
    return fig

def plot_macd(df, ticker, start_date, end_date):
    # Filter data based on selected date range
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    filtered_df = df.loc[mask]
    
    if 'MACD' not in filtered_df.columns or 'Signal_Line' not in filtered_df.columns:
        raise ValueError("MACD and Signal Line data not found in DataFrame.")

    fig = go.Figure()

    # MACD and Signal lines
    fig.add_trace(go.Scatter(
        x=filtered_df.index,
        y=filtered_df['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=filtered_df.index,
        y=filtered_df['Signal_Line'],
        mode='lines',
        name='Signal',
        line=dict(color='orange', width=2)
    ))

    # Add histogram with conditional coloring
    hist_color = np.where((filtered_df['MACD'] - filtered_df['Signal_Line']) >= 0, 'green', 'red')
    fig.add_trace(go.Bar(
        x=filtered_df.index,
        y=filtered_df['MACD'] - filtered_df['Signal_Line'],
        name='Histogram',
        marker_color=hist_color,
        opacity=0.6
    ))

    # Add zero line
    fig.add_hline(y=0, line_width=1, line_color='white')

    fig.update_layout(
        title=f"📉 MACD Indicator - {ticker} ({start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')})",
        template="plotly_dark", 
        height=400,
        xaxis_range=[start_date, end_date],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def plot_rsi(df, ticker, start_date, end_date):
    # Filter data based on selected date range
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    filtered_df = df.loc[mask]
    
    if 'RSI' not in filtered_df.columns:
        raise ValueError("RSI data not found in DataFrame.")

    fig = go.Figure()

    # RSI line
    fig.add_trace(go.Scatter(
        x=filtered_df.index,
        y=filtered_df['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='cyan', width=2)
    ))

    # Add overbought/oversold regions
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor="rgba(255, 0, 0, 0.1)",
        layer="below",
        line_width=0
    )

    fig.add_hrect(
        y0=0, y1=30,
        fillcolor="rgba(0, 255, 0, 0.1)",
        layer="below",
        line_width=0
    )

    # Add threshold lines
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="red",
        annotation_text="Overbought (70)",
        annotation_position="top right"
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="green",
        annotation_text="Oversold (30)",
        annotation_position="bottom right"
    )

    # Add center line
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="gray",
        annotation_text="Midpoint",
        annotation_position="bottom right"
    )

    fig.update_layout(
        title=f'📏 RSI - {ticker} ({start_date.strftime("%d %b %Y")} to {end_date.strftime("%d %b %Y")})',
        yaxis_title='RSI Value',
        yaxis_range=[0, 100],
        xaxis_range=[start_date, end_date],
        height=400,
        template="plotly_dark"
    )

    return fig