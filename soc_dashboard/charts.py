"""Цвета критичности и общие настройки графиков."""
import plotly.graph_objects as go

BLUE = "#888888"
RED = "#FF0000"  # только опасные пропуски / требующее реакции
TEAL = "#707070"
GRAY = "#BBBBBB"


def critical_label(text):
    return f'<span style="color:#FF0000">{text}</span>'


COLORS = {"Инцидент": "#666666", "Ложное срабатывание": "#999999",
          "Другой исход": "#C7C7C7", "Нет вердикта": "#EFEFEF",
          "TP": "#666666", "FN": RED, "FP": "#999999", "TN": "#E8E8E8",
          "Критическая": RED,"Высокая":"#666666","Средняя":"#AAAAAA","Низкая":"#E8E8E8"}
def bar_marker(category,index=0):
    shades=['#666666','#999999','#C7C7C7','#EFEFEF']
    color=COLORS.get(category,shades[index%len(shades)])
    dark=color in ['#666666','#999999']
    return dict(color=color,line=dict(color='#FFFFFF' if dark else '#777777',width=1))


def style(fig, title, x="", y="", height=None):
    # Заголовок находится в панели, а высоту задаёт сетка экрана.
    fig.update_layout(title=None, template="plotly_white", height=None, autosize=True,
                      margin=dict(l=8, r=12, t=23, b=34),
                      font=dict(family="Arial, sans-serif", size=14, color="#454545"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      colorway=[BLUE, TEAL, "#999999", "#BBBBBB"],
                      xaxis_title=x, yaxis_title=y, legend_title_text="",
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font_size=14,
                                  itemsizing="constant", itemwidth=32,
                                  bgcolor="rgba(0,0,0,0)", traceorder="normal"))
    fig.update_xaxes(showgrid=False, automargin=True, title_font_size=14, tickfont_size=14)
    fig.update_yaxes(gridcolor="#EEEEEE", zeroline=False, automargin=True,
                     title_font_size=14, tickfont_size=14)
    return fig


def empty(title, message="В выбранном срезе нет данных"):
    fig = go.Figure()
    fig.add_annotation(text=message, x=.5, y=.5,
                       xref="paper", yref="paper", showarrow=False)
    return style(fig, title)


def horizontal(labels, values, title, unit="Инциденты", color=BLUE):
    if len(labels) == 0:
        return empty(title)
    values=list(values)
    colors=list(color)[::-1] if isinstance(color,(list,tuple)) else color
    fig = go.Figure(go.Bar(x=values[::-1], y=list(labels)[::-1], orientation="h",
                          marker_color=colors,text=[f'{v:,}'.replace(',',' ') for v in values[::-1]],
                          textposition="outside",cliponaxis=False,textfont=dict(color='#454545',size=14)))
    style(fig, title, unit)
    fig.update_xaxes(range=[0,max(max(values)*1.25,1)],rangemode='tozero')
    fig.update_layout(margin_r=50,showlegend=False)
    return fig
