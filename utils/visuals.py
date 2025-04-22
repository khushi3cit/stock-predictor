import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# 1. Line Chart

def plot_candlestick(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing=dict(line=dict(color='#00ff00', width=2)),
        decreasing=dict(line=dict(color='#ff0000', width=2)),
        whiskerwidth=0.8,  # Makes the wicks thicker
        opacity=0.8        # Makes candles more visible
    )])
    
    fig.update_layout(
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        margin=dict(l=20, r=20, t=30, b=20),
        height=600
    )
    return fig

# 2. MACD Chart
def plot_macd(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', name='Signal'))
    fig.update_layout(title="📉 MACD Indicator", template="plotly_white", height=400)
    return fig

# 3. RSI Chart
def plot_rsi(df, period=14):
    fig = px.line(df, x=df.index, y=df['RSI'], title='📏 RSI (Relative Strength Index)', labels={'RSI': 'RSI Value'})
    fig.add_hline(y=70, line_dash="dot", annotation_text="Overbought", line_color="red")
    fig.add_hline(y=30, line_dash="dot", annotation_text="Oversold", line_color="green")
    fig.update_layout(template="plotly_white", height=400)
    return fig

# 4. Column Comparison Chart
def plot_comparison(df, col1, col2):
    fig = px.line(df, x=df.index, y=[col1, col2], title=f'Comparison: {col1} vs {col2}')
    fig.update_layout(template="plotly_white", height=400)
    return fig

# 5. Ichimoku Cloud Plot
def plot_ichimoku(df):
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], 
                            line=dict(color='blue', width=2), 
                            name='Price'))
    
    # Cloud (Span A and Span B)
    fig.add_trace(go.Scatter(x=df.index, y=df['Ichimoku_SpanA'], 
                            fill=None,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False))
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Ichimoku_SpanB'],
                            fill='tonexty',
                            mode='lines',
                            line=dict(width=0),
                            fillcolor='rgba(0,100,80,0.2)',
                            name='Ichimoku Cloud'))
    
    # Conversion and Base lines
    fig.add_trace(go.Scatter(x=df.index, y=df['Ichimoku_Conversion'],
                            line=dict(color='green', width=1),
                            name='Conversion Line'))
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Ichimoku_Base'],
                            line=dict(color='red', width=1),
                            name='Base Line'))
    
    fig.update_layout(title='Ichimoku Cloud', height=500)
    return fig

# 6. Volume + OBV Plot
def plot_volume_analysis(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Price
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Price'), row=1, col=1)
    
    # Volume bars
    colors = ['green' if close > open else 'red' 
              for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
                        marker_color=colors,
                        name='Volume'), row=2, col=1)
    
    # OBV line
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'],
                  line=dict(color='purple', width=2),
                  name='OBV'), row=2, col=1)
    
    fig.update_layout(title='Volume Analysis with OBV',
                     height=600,
                     showlegend=True,
                     xaxis_rangeslider_visible=False)
    return fig
