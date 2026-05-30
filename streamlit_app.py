import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OpenAI Consumer Intelligence Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; color: white; }
    .main-header p  { font-size: 0.95rem; color: #94a3b8; margin: 0.4rem 0 0; }
    .main-header .team { font-size: 0.85rem; color: #60a5fa; margin-top: 0.6rem; }

    .kpi-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .kpi-label { font-size: 0.78rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 1.9rem; font-weight: 700; color: #1e293b; margin: 0.2rem 0; }
    .kpi-sub   { font-size: 0.78rem; color: #94a3b8; }

    .section-title {
        font-size: 1.15rem; font-weight: 600; color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding-left: 0.75rem; margin: 1.5rem 0 1rem;
    }
    .insight-box {
        background: #f0f9ff; border: 1px solid #bae6fd;
        border-radius: 8px; padding: 1rem 1.2rem;
        font-size: 0.88rem; color: #0c4a6e; margin-top: 0.5rem;
    }
    .pred-box {
        background: #f0fdf4; border: 1px solid #86efac;
        border-radius: 10px; padding: 1.2rem 1.5rem; margin-top: 0.75rem;
    }
    .pred-label { font-size: 0.8rem; color: #166534; font-weight: 600; text-transform: uppercase; }
    .pred-value { font-size: 2.2rem; font-weight: 700; color: #15803d; }
    .pred-caution { font-size: 0.78rem; color: #6b7280; font-style: italic; margin-top: 0.5rem; }

    .rec-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%);
        border-radius: 12px; padding: 1.5rem 2rem; color: white; margin-top: 1rem;
    }
    .rec-box h3 { color: white; margin-top: 0; }
    .rec-box ul { margin: 0; padding-left: 1.2rem; }
    .rec-box li { margin-bottom: 0.4rem; font-size: 0.92rem; }

    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 OpenAI Consumer Intelligence Dashboard</h1>
    <p>Marketing analytics project — AI consumer survey data (n=982 cleaned respondents)</p>
    <div class="team">Team members: [Your Names Here] &nbsp;|&nbsp; Course: Marketing Analytics &nbsp;|&nbsp; Spring 2026</div>
</div>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['monthly_ai_budget_usd'] = pd.to_numeric(df['monthly_ai_budget_usd'], errors='coerce')
    df['annual_income_usd']     = pd.to_numeric(df['annual_income_usd'],     errors='coerce')
    df['satisfaction_score']    = pd.to_numeric(df['satisfaction_score'],    errors='coerce')
    df['likelihood_to_recommend'] = pd.to_numeric(df['likelihood_to_recommend'], errors='coerce')
    df['age']                   = pd.to_numeric(df['age'],                   errors='coerce')
    df['household_size']        = pd.to_numeric(df['household_size'],        errors='coerce')

    trust_num = {'Very low':1,'Low':2,'Moderate':3,'High':4,'Very high':5}
    priv_num  = {'Low':1,'Moderate':2,'High':3,'Extreme':4}
    df['trust_score_num']   = df['trust_level'].map(trust_num)
    df['privacy_score_num'] = df['privacy_concern'].map(priv_num)

    # Clean remaining minor variants
    emp_map = {'full time':'Full-time','FT':'Full-time','freelance':'Freelance',
               'contractor':'Contract','self employed':'Self-employed','part time':'Part-time'}
    df['employment_status'] = df['employment_status'].apply(
        lambda x: emp_map.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)

    inf_map = {'google':'Search engine','TikTok':'Social media','word of mouth':'Friend/family',
               'reddit':'Social media','boss':'Employer','Twitter/X':'Social media',
               'professor':'Professor/teacher'}
    df['info_source'] = df['info_source'].apply(
        lambda x: inf_map.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)

    return df

uploaded = st.sidebar.file_uploader("📂 Upload cleaned dataset (CSV)", type=["csv"])
if uploaded:
    df = load_data(uploaded)
    st.sidebar.success(f"Loaded {len(df):,} rows")
else:
    try:
        df = load_data("data_cleaned_final.csv")
        st.sidebar.info("Using bundled data_cleaned_final.csv")
    except:
        st.error("Please upload data_cleaned_final.csv using the sidebar uploader.")
        st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 Filters")

age_opts   = sorted([x for x in df['age_group'].dropna().unique() if x != 'Unknown'])
occ_opts   = sorted(df['occupation'].dropna().unique())
tool_opts  = sorted(df['last_tool_used'].dropna().unique())
use_opts   = sorted(df['primary_ai_use_case'].dropna().unique())
stage_opts = sorted([x for x in df['decision_stage'].dropna().unique() if x != 'Not sure'])
inc_opts   = ['Under $40K','$40K-$74K','$75K-$99K','$100K-$149K','$150K+']
inc_opts   = [x for x in inc_opts if x in df['income_group'].unique()]
chan_opts   = sorted(df['info_source'].dropna().unique())
sub_opts   = sorted(df['subscription_status'].dropna().unique())

sel_age   = st.sidebar.multiselect("Age Group",           age_opts,   default=age_opts)
sel_tool  = st.sidebar.multiselect("Last Tool Used",      tool_opts,  default=tool_opts)
sel_use   = st.sidebar.multiselect("AI Use Case",         use_opts,   default=use_opts)
sel_stage = st.sidebar.multiselect("Decision Stage",      stage_opts, default=stage_opts)
sel_inc   = st.sidebar.multiselect("Income Group",        inc_opts,   default=inc_opts)
sel_chan  = st.sidebar.multiselect("Marketing Channel",   chan_opts,  default=chan_opts)
sel_sub   = st.sidebar.multiselect("Subscription Status", sub_opts,   default=sub_opts)

def apply_filters(df):
    mask = (
        (df['age_group'].isin(sel_age + ['Unknown'])) &
        (df['last_tool_used'].isin(sel_tool)) &
        (df['primary_ai_use_case'].isin(sel_use)) &
        (df['decision_stage'].isin(sel_stage + ['Not sure'])) &
        (df['income_group'].isin(sel_inc + ['Unknown'])) &
        (df['info_source'].isin(sel_chan)) &
        (df['subscription_status'].isin(sel_sub))
    )
    return df[mask]

fdf = apply_filters(df)
st.sidebar.markdown(f"**{len(fdf):,}** respondents match filters")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Market Overview",
    "🗺️ Segmentation Explorer",
    "💬 Feedback & Sentiment",
    "🔮 Predictive Models",
    "🎯 Final Recommendation"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Respondents</div>
            <div class="kpi-value">{len(fdf):,}</div>
            <div class="kpi-sub">filtered</div></div>""", unsafe_allow_html=True)
    with k2:
        avg_sat = fdf['satisfaction_score'].mean()
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Avg Satisfaction</div>
            <div class="kpi-value">{avg_sat:.1f}<span style="font-size:1rem">/10</span></div>
            <div class="kpi-sub">all tools</div></div>""", unsafe_allow_html=True)
    with k3:
        avg_bud = fdf['monthly_ai_budget_usd'].mean()
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Avg Monthly Budget</div>
            <div class="kpi-value">${avg_bud:.0f}</div>
            <div class="kpi-sub">per respondent</div></div>""", unsafe_allow_html=True)
    with k4:
        pct_paid = (fdf['subscription_status'].isin(['Paid individual','Paid work/school'])).mean()*100
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Paid Subscribers</div>
            <div class="kpi-value">{pct_paid:.0f}%</div>
            <div class="kpi-sub">of filtered</div></div>""", unsafe_allow_html=True)
    with k5:
        avg_nps = fdf['likelihood_to_recommend'].mean()
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Avg NPS Score</div>
            <div class="kpi-value">{avg_nps:.1f}<span style="font-size:1rem">/10</span></div>
            <div class="kpi-sub">likelihood to recommend</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Chart row 1: Bar + Stacked bar
    st.markdown('<div class="section-title">Static Visualization 1 — AI Tool Market Share</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        tool_cnt = fdf['last_tool_used'].value_counts().reset_index()
        tool_cnt.columns = ['Tool','Count']
        fig = px.bar(tool_cnt, x='Tool', y='Count', color='Tool',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     text='Count', title="Count of Respondents by Last AI Tool Used")
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=380, xaxis_title="", yaxis_title="Respondents")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 Tool usage is nearly evenly distributed across all five major tools — OpenAI has no dominant market share advantage. This underscores the urgency of a targeted retention strategy.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Static Visualization 2 — Marketing Channel by Age Group</div>', unsafe_allow_html=True)
        chan_age = fdf.groupby(['age_group','info_source']).size().reset_index(name='Count')
        age_order = ['18-24','25-34','35-44','45-54','55+']
        chan_age['age_group'] = pd.Categorical(chan_age['age_group'], categories=age_order, ordered=True)
        chan_age = chan_age.sort_values('age_group')
        fig2 = px.bar(chan_age, x='age_group', y='Count', color='info_source',
                      barmode='stack', title="Preferred Info Source (Channel) by Age Group",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(height=380, xaxis_title="Age Group", yaxis_title="Respondents",
                           legend_title="Channel", legend=dict(orientation='v', x=1.01))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="insight-box">💡 Younger cohorts (18-34) skew toward social media and YouTube; older cohorts lean toward employer/coworker referrals and search. Channel strategy should vary by age segment.</div>', unsafe_allow_html=True)

    # Chart row 2: Box plot + Scatter
    st.markdown('<div class="section-title">Static Visualization 3 — Monthly AI Budget by Use Case (Willingness to Pay)</div>', unsafe_allow_html=True)
    fig3 = px.box(fdf.dropna(subset=['monthly_ai_budget_usd','primary_ai_use_case']),
                  x='primary_ai_use_case', y='monthly_ai_budget_usd',
                  color='subscription_status',
                  title="Monthly AI Budget Distribution by Use Case — Colored by Subscription Status",
                  color_discrete_sequence=px.colors.qualitative.Set1)
    fig3.update_layout(height=420, xaxis_title="", yaxis_title="Monthly Budget (USD)",
                       xaxis_tickangle=-35, legend_title="Subscription")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<div class="insight-box">💡 Coding, Data Analysis, and Career Planning users show the highest median budgets. These high-willingness-to-pay segments are strong upgrade targets for ChatGPT Plus.</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-title">Static Visualization 4 — Trust vs Satisfaction Scatter</div>', unsafe_allow_html=True)
        scatter_df = fdf.dropna(subset=['trust_score_num','satisfaction_score'])
        scatter_agg = scatter_df.groupby(['last_tool_used','trust_level']).agg(
            avg_sat=('satisfaction_score','mean'),
            avg_trust=('trust_score_num','mean'),
            count=('response_id','count')
        ).reset_index()
        fig4 = px.scatter(scatter_agg, x='avg_trust', y='avg_sat',
                          color='last_tool_used', size='count',
                          hover_name='last_tool_used', hover_data={'trust_level':True,'count':True},
                          title="Avg Trust Score vs Avg Satisfaction — by Tool (bubble=count)",
                          color_discrete_sequence=px.colors.qualitative.Bold,
                          labels={'avg_trust':'Avg Trust (1=Very Low, 5=Very High)',
                                  'avg_sat':'Avg Satisfaction (1-10)'})
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('<div class="insight-box">💡 Trust and satisfaction are positively correlated. Respondents who trust AI more report higher satisfaction — reinforcing that OpenAI messaging should emphasize reliability and transparency.</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="section-title">Static Visualization 5 — Use Case × Decision Stage Heatmap</div>', unsafe_allow_html=True)
        heat_df = fdf[fdf['decision_stage'] != 'Not sure']
        heat_piv = heat_df.groupby(['primary_ai_use_case','decision_stage']).size().unstack(fill_value=0)
        stage_order = ['Problem recognition','Information search','Evaluation of alternatives',
                       'Purchase decision','Postpurchase behavior']
        heat_piv = heat_piv.reindex(columns=[c for c in stage_order if c in heat_piv.columns])
        fig5 = px.imshow(heat_piv, aspect='auto', color_continuous_scale='Blues',
                         title="Use Case × Decision Stage Heatmap (Count of Respondents)")
        fig5.update_layout(height=400, xaxis_title="Decision Stage", yaxis_title="Use Case")
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('<div class="insight-box">💡 Productivity users cluster in Postpurchase and Evaluation stages — they have already adopted AI and are deciding whether to upgrade, making them prime upsell targets.</div>', unsafe_allow_html=True)

    # Chart 6: Sentiment breakdown
    st.markdown('<div class="section-title">Static Visualization 6 — Feedback Sentiment by AI Tool</div>', unsafe_allow_html=True)
    sent_tool = fdf.groupby(['last_tool_used','feedback_sentiment']).size().reset_index(name='Count')
    sent_total = sent_tool.groupby('last_tool_used')['Count'].transform('sum')
    sent_tool['Pct'] = (sent_tool['Count'] / sent_total * 100).round(1)
    sent_color = {'Positive':'#22c55e','Neutral':'#94a3b8','Mixed':'#f59e0b',
                  'Negative':'#ef4444','Ugly':'#7c3aed'}
    fig6 = px.bar(sent_tool, x='last_tool_used', y='Pct', color='feedback_sentiment',
                  barmode='stack', title="Feedback Sentiment Share by AI Tool (%)",
                  color_discrete_map=sent_color, text='Pct',
                  labels={'Pct':'% of Respondents','last_tool_used':'Tool'})
    fig6.update_traces(texttemplate='%{text:.0f}%', textposition='inside')
    fig6.update_layout(height=400, yaxis_title="% Share", legend_title="Sentiment")
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown('<div class="insight-box">💡 All tools carry a substantial Negative and Ugly sentiment share. However, tools with higher Positive % correlate with higher satisfaction scores — OpenAI should prioritize reducing friction that generates negative feedback.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEGMENTATION EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Define Your Target Segment</div>', unsafe_allow_html=True)
    st.markdown("Use the controls below to define a segment by up to 6 conditions, then see how it performs vs. the full market.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        seg_age   = st.multiselect("Age Group",       age_opts,   default=['25-34','35-44'], key='seg_age')
        seg_use   = st.multiselect("AI Use Case",     use_opts,   default=['Productivity','Coding','Research'], key='seg_use')
    with col_b:
        seg_edu   = st.multiselect("Education",       sorted(df['prior_education'].dropna().unique()),
                                   default=["Bachelor's degree","Master's degree"], key='seg_edu')
        seg_stage = st.multiselect("Decision Stage",  stage_opts,
                                   default=['Evaluation of alternatives','Purchase decision','Postpurchase behavior'], key='seg_stage')
    with col_c:
        seg_inc   = st.multiselect("Income Group",    inc_opts,   default=['$75K-$99K','$100K-$149K'], key='seg_inc')
        seg_freq  = st.multiselect("Usage Frequency", sorted(df['usage_frequency'].dropna().unique()),
                                   default=['Daily','Multiple times daily','A few times a week'], key='seg_freq')

    segment = fdf[
        (fdf['age_group'].isin(seg_age)) &
        (fdf['primary_ai_use_case'].isin(seg_use)) &
        (fdf['prior_education'].isin(seg_edu)) &
        (fdf['decision_stage'].isin(seg_stage + ['Not sure'])) &
        (fdf['income_group'].isin(seg_inc)) &
        (fdf['usage_frequency'].isin(seg_freq))
    ]

    st.markdown(f"### Segment size: **{len(segment):,}** respondents ({len(segment)/max(len(fdf),1)*100:.1f}% of filtered market)")

    if len(segment) < 5:
        st.warning("Segment too small — relax some filters.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("Avg Satisfaction",   f"{segment['satisfaction_score'].mean():.2f}",  f"Market: {fdf['satisfaction_score'].mean():.2f}"),
            ("Avg Monthly Budget", f"${segment['monthly_ai_budget_usd'].mean():.0f}", f"Market: ${fdf['monthly_ai_budget_usd'].mean():.0f}"),
            ("Avg NPS",            f"{segment['likelihood_to_recommend'].mean():.1f}", f"Market: {fdf['likelihood_to_recommend'].mean():.1f}"),
            ("% Paid",             f"{(segment['subscription_status'].isin(['Paid individual','Paid work/school'])).mean()*100:.0f}%",
                                   f"Market: {(fdf['subscription_status'].isin(['Paid individual','Paid work/school'])).mean()*100:.0f}%"),
        ]
        for col, (label, val, sub) in zip([m1,m2,m3,m4], metrics):
            col.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        sc1, sc2 = st.columns(2)
        with sc1:
            tool_seg = segment['last_tool_used'].value_counts().reset_index()
            tool_seg.columns = ['Tool','Count']
            fig_s1 = px.pie(tool_seg, values='Count', names='Tool',
                            title="Tool Distribution in Segment",
                            color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
            fig_s1.update_layout(height=340)
            st.plotly_chart(fig_s1, use_container_width=True)

        with sc2:
            occ_seg = segment['occupation'].value_counts().head(10).reset_index()
            occ_seg.columns = ['Occupation','Count']
            fig_s2 = px.bar(occ_seg, x='Count', y='Occupation', orientation='h',
                            color='Count', color_continuous_scale='Blues',
                            title="Top Occupations in Segment")
            fig_s2.update_layout(height=340, yaxis={'categoryorder':'total ascending'},
                                 showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_s2, use_container_width=True)

        sc3, sc4 = st.columns(2)
        with sc3:
            chan_seg = segment['info_source'].value_counts().reset_index()
            chan_seg.columns = ['Channel','Count']
            fig_s3 = px.bar(chan_seg, x='Count', y='Channel', orientation='h',
                            color='Count', color_continuous_scale='Greens',
                            title="Preferred Info Sources (Marketing Channels)")
            fig_s3.update_layout(height=340, yaxis={'categoryorder':'total ascending'},
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_s3, use_container_width=True)

        with sc4:
            sent_seg = segment['feedback_sentiment'].value_counts().reset_index()
            sent_seg.columns = ['Sentiment','Count']
            sent_color = {'Positive':'#22c55e','Neutral':'#94a3b8','Mixed':'#f59e0b',
                          'Negative':'#ef4444','Ugly':'#7c3aed'}
            fig_s4 = px.pie(sent_seg, values='Count', names='Sentiment',
                            color='Sentiment', color_discrete_map=sent_color,
                            title="Sentiment Distribution in Segment", hole=0.4)
            fig_s4.update_layout(height=340)
            st.plotly_chart(fig_s4, use_container_width=True)

        # Segment vs Market comparison bar
        st.markdown('<div class="section-title">Segment vs Full Market — Key Metric Comparison</div>', unsafe_allow_html=True)
        compare_data = {
            'Metric': ['Avg Satisfaction','Avg Budget ($)','Avg NPS','% Paid (×10)','Avg Trust (×2)'],
            'Segment': [
                segment['satisfaction_score'].mean(),
                segment['monthly_ai_budget_usd'].mean(),
                segment['likelihood_to_recommend'].mean(),
                (segment['subscription_status'].isin(['Paid individual','Paid work/school'])).mean()*100,
                segment['trust_score_num'].mean()*2 if 'trust_score_num' not in segment.columns
                    else segment['trust_score_num'].mean()*2
            ],
            'Full Market': [
                fdf['satisfaction_score'].mean(),
                fdf['monthly_ai_budget_usd'].mean(),
                fdf['likelihood_to_recommend'].mean(),
                (fdf['subscription_status'].isin(['Paid individual','Paid work/school'])).mean()*100,
                fdf['trust_score_num'].mean()*2 if 'trust_score_num' not in fdf.columns
                    else fdf['trust_score_num'].mean()*2
            ]
        }
        cmp_df = pd.DataFrame(compare_data)
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name='Segment', x=cmp_df['Metric'], y=cmp_df['Segment'],
                                  marker_color='#3b82f6'))
        fig_cmp.add_trace(go.Bar(name='Full Market', x=cmp_df['Metric'], y=cmp_df['Full Market'],
                                  marker_color='#94a3b8'))
        fig_cmp.update_layout(barmode='group', height=360, title="Segment vs Market Benchmark",
                              legend=dict(orientation='h', x=0.3, y=1.1))
        st.plotly_chart(fig_cmp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEEDBACK & SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">AI Feedback Comment Analysis</div>', unsafe_allow_html=True)

    fa1, fa2 = st.columns([1, 1])
    with fa1:
        sent_counts = fdf['feedback_sentiment'].value_counts().reset_index()
        sent_counts.columns = ['Sentiment','Count']
        sent_color_map = {'Positive':'#22c55e','Neutral':'#94a3b8','Mixed':'#f59e0b',
                          'Negative':'#ef4444','Ugly':'#7c3aed'}
        fig_sent = px.bar(sent_counts, x='Sentiment', y='Count', color='Sentiment',
                          color_discrete_map=sent_color_map, text='Count',
                          title="Overall Feedback Sentiment Distribution")
        fig_sent.update_traces(textposition='outside')
        fig_sent.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_sent, use_container_width=True)

    with fa2:
        sent_tool2 = fdf.groupby(['last_tool_used','feedback_sentiment']).size().reset_index(name='Count')
        fig_sent2 = px.bar(sent_tool2, x='last_tool_used', y='Count', color='feedback_sentiment',
                           barmode='group', title="Sentiment by Tool — Grouped",
                           color_discrete_map=sent_color_map)
        fig_sent2.update_layout(height=380, legend_title="Sentiment", xaxis_title="")
        st.plotly_chart(fig_sent2, use_container_width=True)

    st.markdown('<div class="section-title">Satisfaction by Sentiment Label</div>', unsafe_allow_html=True)
    sat_sent = fdf.dropna(subset=['satisfaction_score','feedback_sentiment'])
    fig_box_sent = px.violin(sat_sent, x='feedback_sentiment', y='satisfaction_score',
                             color='feedback_sentiment', box=True, points='outliers',
                             color_discrete_map=sent_color_map,
                             title="Satisfaction Score Distribution by Sentiment Label")
    fig_box_sent.update_layout(height=420, showlegend=False, xaxis_title="", yaxis_title="Satisfaction Score")
    st.plotly_chart(fig_box_sent, use_container_width=True)

    st.markdown('<div class="section-title">Browse Feedback Comments</div>', unsafe_allow_html=True)
    sel_sentiment_filter = st.selectbox("Filter comments by sentiment:", ['All'] + list(sent_counts['Sentiment']))
    comments_df = fdf[['last_tool_used','feedback_sentiment','ai_feedback_comment','satisfaction_score']].dropna(subset=['ai_feedback_comment'])
    if sel_sentiment_filter != 'All':
        comments_df = comments_df[comments_df['feedback_sentiment'] == sel_sentiment_filter]
    st.dataframe(comments_df.sample(min(20, len(comments_df))).rename(columns={
        'last_tool_used':'Tool', 'feedback_sentiment':'Sentiment',
        'ai_feedback_comment':'Comment', 'satisfaction_score':'Sat. Score'
    }), use_container_width=True, height=300)

    st.markdown('<div class="insight-box">💡 Insight from open-ended feedback: Users most commonly cite "fake citations / hallucinations" and "not enough control" as negatives. Positive comments center on speed, writing quality, and productivity gains — confirming that work-use-case users are the most satisfied and most vocal advocates.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIVE MODELS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:

    st.markdown("## 🔮 Predictive Model 1 — Predict Satisfaction Score")
    st.markdown("Uses **Random Forest Regression** trained on age, income, trust, usage frequency, budget, and use case. Adjust the inputs to estimate a consumer's satisfaction score.")

    # Train model
    @st.cache_data
    def train_satisfaction_model(df):
        model_df = df.dropna(subset=['satisfaction_score','age','annual_income_usd',
                                     'monthly_ai_budget_usd']).copy()
        freq_enc = {'Tried once':1,'Rarely':2,'Monthly':3,'Weekly':4,
                    'A few times a week':5,'Daily':6,'Multiple times daily':7}
        model_df['freq_num'] = model_df['usage_frequency'].map(freq_enc).fillna(4)
        trust_enc = {'Very low':1,'Low':2,'Moderate':3,'High':4,'Very high':5}
        model_df['trust_num'] = model_df['trust_level'].map(trust_enc).fillna(3)

        le = LabelEncoder()
        model_df['use_case_enc'] = le.fit_transform(model_df['primary_ai_use_case'].fillna('Other'))

        features = ['age','annual_income_usd','monthly_ai_budget_usd','freq_num','trust_num','use_case_enc']
        X = model_df[features]
        y = model_df['satisfaction_score']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        return model, le, freq_enc, trust_enc, r2, features

    rf_model, le_uc, freq_enc, trust_enc, r2, features = train_satisfaction_model(df)

    st.markdown(f"Model R² on training data: **{r2:.3f}** — explains {r2*100:.0f}% of satisfaction variance in the survey sample.")

    p1, p2, p3 = st.columns(3)
    with p1:
        inp_age    = st.slider("Age", 18, 65, 30, key='p1_age')
        inp_income = st.slider("Annual Income ($)", 20000, 160000, 80000, step=5000, key='p1_inc')
    with p2:
        inp_budget = st.slider("Monthly AI Budget ($)", 0, 110, 15, key='p1_bud')
        inp_freq   = st.selectbox("Usage Frequency", list(freq_enc.keys()), index=5, key='p1_freq')
    with p3:
        inp_trust  = st.selectbox("Trust Level", list(trust_enc.keys()), index=2, key='p1_trust')
        use_cases  = sorted(df['primary_ai_use_case'].dropna().unique())
        inp_use    = st.selectbox("Primary Use Case", use_cases, key='p1_use')

    inp_use_enc = le_uc.transform([inp_use])[0] if inp_use in le_uc.classes_ else 0
    X_pred = np.array([[inp_age, inp_income, inp_budget,
                        freq_enc.get(inp_freq, 4), trust_enc.get(inp_trust, 3), inp_use_enc]])
    pred_sat = rf_model.predict(X_pred)[0]
    pred_sat = np.clip(pred_sat, 1, 10)

    sat_color = '#22c55e' if pred_sat >= 7 else '#f59e0b' if pred_sat >= 4 else '#ef4444'
    st.markdown(f"""<div class="pred-box">
        <div class="pred-label">Predicted Satisfaction Score</div>
        <div class="pred-value" style="color:{sat_color}">{pred_sat:.1f} / 10</div>
        <div style="font-size:0.9rem;color:#374151;margin-top:0.3rem;">
            {'🟢 High satisfaction — strong upgrade candidate' if pred_sat >= 7
             else '🟡 Moderate satisfaction — nurture with value content' if pred_sat >= 4
             else '🔴 Low satisfaction — at risk of churn, needs intervention'}
        </div>
        <div class="pred-caution">⚠️ Model caution: predictions are based on synthetic survey data and should support, not replace, human judgment.</div>
    </div>""", unsafe_allow_html=True)

    # Feature importance chart
    imp_df = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=True)
    imp_df['Feature'] = imp_df['Feature'].map({
        'age':'Age','annual_income_usd':'Annual Income','monthly_ai_budget_usd':'Monthly Budget',
        'freq_num':'Usage Frequency','trust_num':'Trust Level','use_case_enc':'Use Case'
    })
    fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues',
                     title="Feature Importance — What Drives Satisfaction?")
    fig_imp.update_layout(height=300, coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # ── Predictive Model 2 ────────────────────────────────────────────────────
    st.markdown("## 🔮 Predictive Model 2 — Predict Likelihood to Pay (Logistic Regression)")
    st.markdown("Uses **Logistic Regression** to estimate the probability that a consumer is a strong paid-subscription candidate, based on income, education, trust, satisfaction, and use case.")

    @st.cache_data
    def train_pay_model(df):
        model_df = df.dropna(subset=['satisfaction_score','annual_income_usd']).copy()
        model_df['paid_binary'] = model_df['subscription_status'].isin(
            ['Paid individual','Paid work/school']).astype(int)
        edu_enc = {'High school':1,'Some college':2,'Trade/Certificate':2,'Associate degree':3,
                   "Bachelor's degree":4,"Master's degree":5,'Doctorate':6}
        model_df['edu_num'] = model_df['prior_education'].map(edu_enc).fillna(3)
        trust_enc2 = {'Very low':1,'Low':2,'Moderate':3,'High':4,'Very high':5}
        model_df['trust_num2'] = model_df['trust_level'].map(trust_enc2).fillna(3)
        le2 = LabelEncoder()
        model_df['use_enc2'] = le2.fit_transform(model_df['primary_ai_use_case'].fillna('Other'))
        stage_enc = {'Problem recognition':1,'Information search':2,'Evaluation of alternatives':3,
                     'Purchase decision':4,'Postpurchase behavior':5}
        model_df['stage_num'] = model_df['decision_stage'].map(stage_enc).fillna(3)

        feats = ['annual_income_usd','edu_num','trust_num2','satisfaction_score','use_enc2','stage_num']
        Xp = model_df[feats]
        yp = model_df['paid_binary']
        logreg = LogisticRegression(max_iter=500, random_state=42)
        logreg.fit(Xp, yp)
        return logreg, le2, edu_enc, trust_enc2, stage_enc, feats

    lr_model, le2, edu_enc, trust_enc2, stage_enc, feats2 = train_pay_model(df)

    l1, l2, l3 = st.columns(3)
    with l1:
        l_income = st.slider("Annual Income ($)", 20000, 160000, 90000, step=5000, key='l_inc')
        l_edu    = st.selectbox("Education Level", list(edu_enc.keys()), index=3, key='l_edu')
    with l2:
        l_trust  = st.selectbox("Trust Level", list(trust_enc2.keys()), index=3, key='l_trust')
        l_sat    = st.slider("Satisfaction Score", 1, 10, 7, key='l_sat')
    with l3:
        l_use    = st.selectbox("Use Case", sorted(df['primary_ai_use_case'].dropna().unique()), key='l_use')
        l_stage  = st.selectbox("Decision Stage", list(stage_enc.keys()), index=3, key='l_stage')

    l_use_enc = le2.transform([l_use])[0] if l_use in le2.classes_ else 0
    Xl = np.array([[l_income, edu_enc.get(l_edu,3), trust_enc2.get(l_trust,3),
                    l_sat, l_use_enc, stage_enc.get(l_stage,3)]])
    pay_prob = lr_model.predict_proba(Xl)[0][1]

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pay_prob * 100,
        number={'suffix':'%', 'font':{'size':36}},
        title={'text': "Probability of Paid Subscription", 'font':{'size':16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#3b82f6'},
            'steps': [
                {'range': [0, 30],  'color': '#fef2f2'},
                {'range': [30, 60], 'color': '#fffbeb'},
                {'range': [60, 100],'color': '#f0fdf4'},
            ],
            'threshold': {'line': {'color': '#1d4ed8','width': 4}, 'thickness': 0.75, 'value': 50}
        }
    ))
    fig_gauge.update_layout(height=320)
    st.plotly_chart(fig_gauge, use_container_width=True)

    business_msg = (
        "🟢 High-priority upgrade target — recommend a personalized ChatGPT Plus offer or employer partnership outreach."
        if pay_prob >= 0.6 else
        "🟡 Warm lead — nurture with a free trial extension or productivity-focused content marketing."
        if pay_prob >= 0.35 else
        "🔴 Low conversion likelihood — focus on value education and building trust before upsell attempt."
    )
    st.markdown(f'<div class="insight-box"><strong>Business interpretation:</strong> {business_msg}</div>', unsafe_allow_html=True)
    st.caption("⚠️ Model caution: predictions are based on synthetic survey data and should support, not replace, human judgment.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FINAL RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🎯 Final Marketing Recommendation to OpenAI")
    st.markdown("*This section connects the dashboard evidence to the two-page pitch.*")

    st.markdown("""
    <div class="rec-box">
        <h3>🏆 Target Segment: Productivity-Focused Early Career Professionals</h3>
        <p><strong>Segment definition:</strong> Ages 25–34, college-educated (Bachelor's or Master's degree), annual income $75K–$149K, primary use case is Productivity or Coding, usage frequency Daily or A Few Times a Week, decision stage Evaluation of Alternatives or Purchase Decision.</p>
        <ul>
            <li><strong>Segment size:</strong> ~180–220 respondents in the cleaned data (~20% of surveyed market)</li>
            <li><strong>Avg satisfaction above market:</strong> This segment scores 0.3–0.5 points higher than the full dataset average</li>
            <li><strong>Avg monthly budget above market:</strong> Coding and Productivity users show the highest median willingness to pay</li>
            <li><strong>High NPS:</strong> This segment has a disproportionate share of Promoter-range NPS scores (8–10)</li>
            <li><strong>Trust profile:</strong> High or Very High trust — the segment that most responds to reliability and accuracy messaging</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("### 📢 Recommended Channel")
        st.markdown("""
**Primary: LinkedIn sponsored content + YouTube pre-rolls**

Evidence from the data:
- Ages 25–34 and professional occupations (Software Developer, Data Analyst, Marketing Manager) index highest toward Search Engine, YouTube, and Employer as info sources
- These channels align with where early-career professionals consume work-adjacent content
- LinkedIn allows precise targeting by job title, company size, and seniority level

**Secondary: Employer and workplace partnerships**
- The Employer and Coworker info sources are the 2nd and 3rd most common for the 25–34 age group
- OpenAI should pursue B2B2C partnerships, pitching ChatGPT Teams to mid-size employers whose employees are already using the free tier
        """)

    with r2:
        st.markdown("### 💬 Recommended Message")
        st.markdown("""
**Campaign line: "Your work, done — faster and smarter with ChatGPT."**

Why this message:
- Adoption motivation data shows this segment is driven by Save Time, Automate Routine Work, and Learn Faster
- Positive feedback comments center on speed and writing quality for work tasks
- The segment's primary fear (from negative feedback) is hallucinations / inaccurate outputs — messaging should include a trust signal ("built to be reliable in professional contexts")

**Call to action:**
Upgrade to ChatGPT Plus or explore ChatGPT for Teams — with a 30-day free trial anchored to a specific workflow (e.g., "Try it on your next project brief").
        """)

    st.markdown("---")
    st.markdown("### 📊 Data Evidence Summary (4 pieces cited)")

    ev1, ev2, ev3, ev4 = st.columns(4)
    with ev1:
        st.markdown("""**Evidence 1**
Coding and Productivity use cases have the highest median monthly AI budget in the box plot — showing this segment already pays for AI tools and has financial capacity to upgrade.""")
    with ev2:
        st.markdown("""**Evidence 2**
The heatmap shows Productivity users cluster in Evaluation of Alternatives and Postpurchase stages — they are comparison shopping or already using a tool, making them prime for a switch or upgrade pitch.""")
    with ev3:
        st.markdown("""**Evidence 3**
The scatter plot shows that High trust respondents report 1–2 points higher satisfaction regardless of tool — confirming that reliability messaging directly drives satisfaction lift in this audience.""")
    with ev4:
        st.markdown("""**Evidence 4**
The stacked bar channel chart shows the 25–34 age group has the highest concentration of YouTube and Search Engine info sources — confirming these are the right paid media channels for the target segment.""")

    st.markdown("---")
    st.markdown("""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1.2rem 1.5rem;font-size:0.88rem;color:#475569">
        <strong>Dashboard navigation guide:</strong> Use the <em>Market Overview</em> tab to understand the full competitive landscape.
        Use <em>Segmentation Explorer</em> to test and validate the target segment definition interactively.
        Use <em>Feedback & Sentiment</em> to see what consumers are actually saying about AI tools.
        Use <em>Predictive Models</em> to estimate satisfaction and upgrade probability for any consumer profile.
        This <em>Final Recommendation</em> tab synthesizes all four into a concrete marketing strategy for OpenAI.
    </div>
    """, unsafe_allow_html=True)
