"""
RYP Season Dashboard — Interactive NFL Picks Analysis
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
import pickle

from ml_model import ALL_TEAMS, predict_pick, load_models
from data_prep import PLAYERS

st.set_page_config(page_title="RYP 2025-2026 Season Wrapped", layout="wide")
st.title("RYP 2025-2026 Season Wrapped")

# ── Load precomputed data (instant) ─────────────────────────────────────────

@st.cache_data
def get_cache():
    with open("cache.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def get_models():
    return load_models()

c = get_cache()
models, feature_names = get_models()

# conference lookup for scenario generation
team_meta = c["teams_df"][["team_id", "team_conference", "team_division"]].dropna(subset=["team_division"]).drop_duplicates(subset=["team_id"])
conf_map = team_meta.set_index("team_id")["team_conference"].to_dict()

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab0, tab1, tab2, tab3 = st.tabs([
    "Summary",
    "Team Performance",
    "Personal Bias & Patterns",
    "Pick Predictor",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0: SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
with tab0:
    st.header("Season Summary")

    summary_text_col, summary_img_col = st.columns([2, 1])
    with summary_text_col:
        st.markdown(
            "Happy Super Bowl Sunday gentlemen, and welcome to RYP 2025-26 Wrapped! \n\n"
            "7 long, cold, dark years but the NEW ENGLAND PATRIOTS ARE BACK IN THE SUPER BOWL!\n\n "
            "And we have our own Super Bowl left to settle the league winner this year."
            " **Exciting Whites** and **Yianni** are heading into the final game tied at **174 points**. "
            "This is the first time the season has come down to the final game since our "
            "2022-23 season. Hope you boys pick different teams!\n\n"
            "Either way, we crown a new league winner: standout rookie **Exciting Whites** or **Yianni**.\n\n"
            "The race was tight all season — **5 different leaders** across 18 weeks and "
            "**10 different managers** posted the highest weekly score at some point.\n\n"
            "Lots of shoutouts to **Exciting Whites** this season — one-hit wonder with "
            "**15 points in a single week** (that's 14 of 16 correct picks, a **1 in 478** long shot "
            "that may never be broken), regular season co-winner with **ADon**, "
            "and tied for the lead heading into the Super Bowl.\n\n"
            "And finally, huge shoutout to our commish **Derek** 👏👏👏 for another great season, "
            "the fun side games, and world-class commitment to keeping this league exciting every week. 🙌🎉\n\n"
            "OK let's get to the DATA."
        )
    with summary_img_col:
        st.image(
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ0hbAYWra9W5dti-1vZCV6UM50hPr5dB52BQ&s",
            use_container_width=True,
        )

    st.subheader("How to Navigate the Analysis!!")
    st.markdown(
        """
        <div style="display:flex; align-items:stretch; justify-content:center; gap:0; margin:1.5em 0;">
            <div style="flex:1; border:1px solid #ccc; border-radius:8px; padding:18px 14px; text-align:center;">
                <div style="font-size:0.75em; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#636EFA;">01</div>
                <div style="font-size:1.05em; font-weight:bold; color:#222; margin-top:6px;">Summary (we're here)</div>
                <div style="font-size:0.82em; color:#555; margin-top:8px;">Season standings & the state of the race heading into the Super Bowl</div>
            </div>
            <div style="display:flex; align-items:center; font-size:1.2em; color:#aaa; padding:0 8px;">&#9654;</div>
            <div style="flex:1; border:1px solid #ccc; border-radius:8px; padding:18px 14px; text-align:center;">
                <div style="font-size:0.75em; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#00996e;">02</div>
                <div style="font-size:1.05em; font-weight:bold; color:#222; margin-top:6px;">Team Performance</div>
                <div style="font-size:0.82em; color:#555; margin-top:8px;">How NFL teams performed ATS and straight-up — who covered and who burned you</div>
            </div>
            <div style="display:flex; align-items:center; font-size:1.2em; color:#aaa; padding:0 8px;">&#9654;</div>
            <div style="flex:1; border:1px solid #ccc; border-radius:8px; padding:18px 14px; text-align:center;">
                <div style="font-size:0.75em; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#8a3fcc;">03</div>
                <div style="font-size:1.05em; font-weight:bold; color:#222; margin-top:6px;">Bias & Patterns</div>
                <div style="font-size:0.82em; color:#555; margin-top:8px;">Your subconscious favorites, herd mentality, and blind spots revealed</div>
            </div>
            <div style="display:flex; align-items:center; font-size:1.2em; color:#aaa; padding:0 8px;">&#9654;</div>
            <div style="flex:1; border:1px solid #ccc; border-radius:8px; padding:18px 14px; text-align:center;">
                <div style="font-size:0.75em; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#d4700a;">04</div>
                <div style="font-size:1.05em; font-weight:bold; color:#222; margin-top:6px;">Pick Predictor</div>
                <div style="font-size:0.82em; color:#555; margin-top:8px;">ML model trained to think like you — generate a game and see if it nails your pick</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Current RYP Standings")

    standings_df = pd.DataFrame([
        {"Rank": 1, "Manager": "Exciting Whites", "W": 157, "L": 127, "Pts": 174},
        {"Rank": 2, "Manager": "Yianni", "W": 151, "L": 133, "Pts": 174},
        {"Rank": 3, "Manager": "b_hop", "W": 154, "L": 130, "Pts": 168},
        {"Rank": 4, "Manager": "ADon", "W": 150, "L": 134, "Pts": 166},
        {"Rank": 5, "Manager": "MC$", "W": 145, "L": 139, "Pts": 165},
        {"Rank": 5, "Manager": "Maye Magic", "W": 141, "L": 141, "Pts": 165},
        {"Rank": 7, "Manager": "derelicious", "W": 151, "L": 133, "Pts": 164},
        {"Rank": 7, "Manager": "Willheser", "W": 144, "L": 140, "Pts": 164},
        {"Rank": 9, "Manager": "Kevin", "W": 146, "L": 134, "Pts": 161},
        {"Rank": 9, "Manager": "Vegas", "W": 143, "L": 141, "Pts": 161},
        {"Rank": 11, "Manager": "mrmcwinnerson", "W": 142, "L": 142, "Pts": 157},
        {"Rank": 12, "Manager": "P-Otys", "W": 135, "L": 149, "Pts": 154},
        {"Rank": 13, "Manager": "Ripw1124", "W": 114, "L": 112, "Pts": 119},
    ])
    standings_df = standings_df.set_index("Rank")

    st.dataframe(standings_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: TEAM PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Team Performance")
    st.caption("How did NFL teams perform this season?")

    # ── Straight-Up (Moneyline) Record ──────────────────────────────────────
    ml = c["ml"].copy()
    ml["_priority"] = ml["team"].map(lambda t: 0 if t == "NE" else 1)
    ml = ml.sort_values(["ml_wins", "_priority", "ml_pct"], ascending=[False, True, False]).reset_index(drop=True)
    ml = ml.drop("_priority", axis=1)
    ml["wins_label"] = ml["ml_wins"].astype(int).astype(str)

    prose_col, chart_col = st.columns([1, 2])
    with prose_col:
        st.subheader("Straight-Up Record")
        top = ml.iloc[0]
        st.markdown(
            f"Drake Maye led the New England Patriots to a NFL T-1 **{int(top['ml_wins'])} wins**. "
            f"Denver and Seattle had admirable seasons but stand no chance to "
            f"Drake \"Drake 'The Schedule' Maye\" Maye.\n\n"
            f"And shoutout to **NYJ** for proudly anchoring the bottom of the standings. "
            f"Not a single receiver hit 400 yards on the season: a feat that would take roughly 22 yards a game to achieve. "
            f"Their defense also recorded 0 interceptions, which has literally never happened before in NFL history."
        )
    with chart_col:
        fig_ml = px.bar(
            ml, y="team", x="ml_pct", orientation="h",
            color="ml_pct", color_continuous_scale="RdYlGn",
            labels={"ml_pct": "Win %", "team": ""},
            text="wins_label",
        )
        fig_ml.add_vline(x=0.5, line_dash="dash", line_color="gray")
        fig_ml.update_traces(textposition="outside")
        fig_ml.update_layout(
            height=700,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            xaxis=dict(tickformat=".0%"),
            margin=dict(t=10),
        )
        st.plotly_chart(fig_ml, use_container_width=True)

    # ── ATS Record by Team ──────────────────────────────────────────────────
    ats = c["ats"].copy()
    ats["covers_label"] = ats["ats_wins"].astype(int).astype(str)

    prose_col, chart_col = st.columns([1, 2])
    with prose_col:
        st.subheader("Against the Spread")
        top_ats = ats.iloc[0]
        bot_ats = ats.iloc[-1]
        st.markdown(
            f"Now for the chart that matters for our league's ATS rules.\n\n"
            f"ATS is the name of the game in RYP. Winning outright means nothing "
            f"if the team doesn't cover. The key to winning the league? "
            f"Capitalize on teams that consistently beat the spread and stay far "
            f"away from the ones that don't.\n\n"
            f"**{top_ats['team']}** led the way with **{int(top_ats['ats_wins'])} covers**, "
            f"while **{bot_ats['team']}** brought up the rear at just "
            f"**{int(bot_ats['ats_wins'])}**.\n\n"
            f"Remember this. TB with just **{int(bot_ats['ats_wins'])} covers**. This will come up again later.\n\n"
            f"And also remember when Baker was the MVP front runner around Week 7 or so?"
        )
    with chart_col:
        fig_ats = px.bar(
            ats, y="team", x="ats_pct", orientation="h",
            color="ats_pct", color_continuous_scale="RdYlGn",
            labels={"ats_pct": "ATS Win %", "team": ""},
            hover_data={"ats_wins": True, "ats_losses": True, "ats_pushes": True, "covers_label": False},
            text="covers_label",
        )
        fig_ats.add_vline(x=0.5, line_dash="dash", line_color="gray")
        fig_ats.update_traces(textposition="outside")
        fig_ats.update_layout(
            height=700,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            xaxis=dict(tickformat=".0%"),
            margin=dict(t=10),
        )
        st.plotly_chart(fig_ats, use_container_width=True)

    # ── Home vs Away ATS Cover Rate ─────────────────────────────────────────
    ha = c["ha"].sort_values("home_cover_pct", ascending=False)
    avg_home = ha["home_cover_pct"].mean()
    avg_away = ha["away_cover_pct"].mean()

    prose_col, chart_col = st.columns([1, 2])
    with prose_col:
        st.subheader("Home vs Away ATS Cover Rate")
        st.markdown(
            f"Does home field advantage still mean anything?\n\n"
            f"On average, home teams covered the spread **{avg_home:.0%}** of the time "
            f"this season, while away teams covered **{avg_away:.0%}**.\n\n"
            f"Foxborough, Arrowhead, Lumen Field: supposed to be nightmares for visiting "
            f"teams. The data says otherwise. Don't let the stadium fool you.\n\n"
            f"Jacksonville had them all beat. Must be the jacuzzi in Section 209."
        )
    with chart_col:
        fig_ha = go.Figure()
        fig_ha.add_trace(go.Bar(
            name="Home", y=ha["team"], x=ha["home_cover_pct"],
            orientation="h", marker_color="#636EFA",
        ))
        fig_ha.add_trace(go.Bar(
            name="Away", y=ha["team"], x=ha["away_cover_pct"],
            orientation="h", marker_color="#EF553B",
        ))
        fig_ha.add_vline(x=0.5, line_dash="dash", line_color="gray")
        fig_ha.update_layout(
            barmode="group", xaxis_title="Cover Rate", height=700,
            xaxis=dict(tickformat=".0%"),
            margin=dict(t=10),
        )
        st.plotly_chart(fig_ha, use_container_width=True)

    # ── Spread Impact on Favorites ──────────────────────────────────────────
    si = c["si"]

    prose_col, chart_col = st.columns([1, 2])
    with prose_col:
        st.subheader("Spread Impact on Favorites")
        st.markdown(
            f"At a 0.5-point spread, we're essentially just picking a winner: "
            f"and this season the favorite was likely to cover.\n\n"
            f"But bump it to 1.5-3 points and the underdog is actually more "
            f"likely to cover the spread.\n\n"
            f"The number inside each bar is the sample size (total games in that bucket)."
        )
    with chart_col:
        fig_si = px.bar(
            si, x="spread_bucket", y="fav_cover_rate",
            text="games",
            labels={"fav_cover_rate": "Favorite Cover Rate", "spread_bucket": "Spread Range"},
            color="fav_cover_rate", color_continuous_scale="RdYlGn",
        )
        fig_si.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig_si.update_layout(
            height=450, coloraxis_showscale=False,
            yaxis=dict(tickformat=".0%"),
            margin=dict(t=10),
        )
        st.plotly_chart(fig_si, use_container_width=True)

    # ── Favorites Coverage by Week ──────────────────────────────────────────
    ws = c["ws"].copy()
    ws["fav_covered"] = (ws["fav_cover_rate"] * ws["games"]).round().astype(int)
    ws["label"] = ws["fav_covered"].astype(str) + "/" + ws["games"].astype(str)

    prose_col, chart_col = st.columns([1, 2])
    with prose_col:
        st.subheader("Favorites Coverage by Week")
        best_wk = ws.loc[ws["fav_cover_rate"].idxmax()]
        worst_wk = ws.loc[ws["fav_cover_rate"].idxmin()]
        st.markdown(
            f"For every game each week, one team is favored to win. "
            f"This chart shows how often that favorite actually covered the spread.\n\n"
            f"Each bar is labeled X/Y: X favorites covered out of Y total games that week.\n\n"
            f"**Week {int(best_wk['week'])}** was the most predictable "
            f"({int(best_wk['fav_covered'])}/{int(best_wk['games'])}), "
            f"while **Week {int(worst_wk['week'])}** was upset city "
            f"({int(worst_wk['fav_covered'])}/{int(worst_wk['games'])})."
        )
    with chart_col:
        fig_ws = px.bar(
            ws, x="week", y="fav_cover_rate",
            labels={"fav_cover_rate": "Favorite Cover Rate", "week": "Week"},
            color="fav_cover_rate", color_continuous_scale="RdYlGn",
            text="label",
        )
        fig_ws.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig_ws.update_layout(
            coloraxis_showscale=False,
            yaxis=dict(tickformat=".0%"),
            margin=dict(t=10),
        )
        st.plotly_chart(fig_ws, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: BIAS & PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Bias & Patterns")
    st.caption("This is where things get interesting: our group and individual tendencies revealed.")

    st.info(
        "**Data Notes:**\n"
        "- ~2.6% of picks were adjusted for CFB, so not all picks reflect true sentiment\n"
        "- Ripw1124 checked out late season (similar to the Jets), so that data is imperfect\n"
        "- This analysis does not account for double downs: we don't want to drown the picks with double pick weighting"
    )

    # ── Quiz ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Quick quiz before we dive in:**")
    quiz_answer = st.radio(
        "Which team did our group pick the most collectively this season?",
        ["NE", "BUF", "SEA", "TB"],
        index=None,
        horizontal=True,
    )
    if quiz_answer is not None:
        mpt_quiz = c["mpt"].reset_index(drop=True)
        tb_row = mpt_quiz[mpt_quiz["pick"] == "TB"]
        tb_picks = int(tb_row["total_picks"].values[0])
        if quiz_answer == "TB":
            st.success(
                f"You got it! **TB** was our #1 most picked team this season with **{tb_picks} selections**. "
                f"Riding the coattails of last year's run and the swagger of Baker, we collectively picked "
                f"the team that finished **dead last in ATS** the most. That is some epic house-always-wins energy."
            )
        else:
            st.error(
                f"Somehow... it was **TB**: our #1 most picked team this season with **{tb_picks} selections**. "
                f"Riding the coattails of last year's run and the swagger of Baker, we collectively picked "
                f"the team that finished **dead last in ATS** the most. That is some epic house-always-wins energy."
            )
    st.markdown("---")

    # ── Most Picked Teams ───────────────────────────────────────────────────
    mpt = c["mpt"]

    prose_col, chart_col = st.columns([1, 2])
    with prose_col:
        st.subheader("Most Picked Teams")
        top_team = mpt.iloc[0]
        second_team = mpt.iloc[1]
        st.markdown(
            f"🥁🥁🥁 Across 12 managers + Vegas, **{top_team['pick']}** was the most popular pick "
            f"this season with **{int(top_team['total_picks'])} selections**, followed by "
            f"**{second_team['pick']}** at **{int(second_team['total_picks'])}**.\n\n"
            f"I'm still trying to wrap my head around how we collectively picked the worst performing team the most this season."
        )
    with chart_col:
        fig_mpt = px.bar(
            mpt, x="pick", y="total_picks",
            labels={"pick": "Team", "total_picks": "Times Picked"},
            color="total_picks", color_continuous_scale="Blues",
        )
        fig_mpt.update_layout(coloraxis_showscale=False, margin=dict(t=10))
        st.plotly_chart(fig_mpt, use_container_width=True)

    # ── Favorite Pick Rate ──────────────────────────────────────────────────
    fur = c["fur"]

    prose_col, chart_col = st.columns([1, 2])
    with prose_col:
        st.subheader("Picking the Favorite")
        top_fav = fur.iloc[0]
        bot_fav = fur.iloc[-1]
        st.markdown(
            f"How often does each manager go with the odds-on favorite?\n\n"
            f"**{top_fav['player']}** leads the pack, picking the favorite "
            f"**{top_fav['fav_rate']:.0%}** of the time. That guy is no fun!\n\n"
            f"On the other end, "
            f"**{bot_fav['player']}** trusted the underdog more than anyone.\n\n"
            f"The 50% line is your gut check: above it, you trust the oddsmakers. "
            f"Below it, you trust chaos. "
            f"P-Otys should now be referred to as P-Chaos."
        )
    with chart_col:
        fig_fur = px.bar(
            fur, y="player", x="fav_rate", orientation="h",
            labels={"fav_rate": "% Picking Favorite", "player": ""},
            color="fav_rate", color_continuous_scale="Oranges",
        )
        fig_fur.add_vline(x=0.5, line_dash="dash", line_color="gray")
        fig_fur.update_layout(
            height=500, yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False, margin=dict(t=10),
        )
        fig_fur.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig_fur, use_container_width=True)

    # ── Herd Mentality ──────────────────────────────────────────────────────
    hm = c["herd"]

    prose_col, chart_col = st.columns([1, 2])
    with prose_col:
        st.subheader("Herd Mentality")
        bot_herd = hm.iloc[-1]
        st.markdown(
            f"How often does each manager agree with the majority pick?\n\n"
            f"You can't even really see it but **{bot_herd['player']}** is the lone wolf of the year: "
            f"with a substantially anti-herd mentality selection at just **{bot_herd['herd_rate']:.0%}**. "
            f"It didn't really work out this season but it's definitely how to separate yourself from the pack."
        )
    with chart_col:
        fig_hm = px.bar(
            hm, y="player", x="herd_rate", orientation="h",
            labels={"herd_rate": "% With Majority", "player": ""},
            color="herd_rate", color_continuous_scale="Purples",
        )
        fig_hm.update_layout(
            height=500, yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False, margin=dict(t=10),
        )
        fig_hm.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig_hm, use_container_width=True)

    # ── Picks Above Average (PAA) ──────────────────────────────────────────
    st.subheader("Picks Above Average (PAA)")
    paa = c["paa"]

    paa_stacked = paa.stack().reset_index()
    paa_stacked.columns = ["Player", "Team", "PAA"]
    paa_stacked["PAA"] = paa_stacked["PAA"].round(1)
    paa_filtered = paa_stacked[~paa_stacked["Player"].isin(["Vegas", "Ripw1124"])]
    top5_paa = paa_filtered.nlargest(5, "PAA")[["Player", "Team", "PAA"]].reset_index(drop=True)
    bot5_paa = paa_filtered.nsmallest(5, "PAA")[["Player", "Team", "PAA"]].reset_index(drop=True)

    st.markdown(
        "Which teams did we lean into and which ones did we avoid? "
        "This heatmap shows each manager's pick count vs the league average: "
        "green means you picked that team more than everyone else, red means you stayed away.\n\n"
        "**Kevin**, historically a Packers over-indexer, cracks the top 5 this year but for the **Chicago Bears**. "
        "Incredible. Ben Johnson can sway heads.\n\n"
        "**mrmcwinnerson** didn't like the Giants this year. Alpha mentality, I don't like them any year.\n\n"
        "**Willheser**, coming off a spectacular rookie campaign, saw some regression to the mean this year. "
        "But he was right to go against KC, they had an uncharacteristically poor season."
    )

    tbl1, tbl2 = st.columns(2)
    with tbl1:
        st.markdown("**Biggest Fans (Top 5 PAA)**")
        st.dataframe(top5_paa, use_container_width=True, hide_index=True)
    with tbl2:
        st.markdown("**Biggest Haters (Bottom 5 PAA)**")
        st.dataframe(bot5_paa, use_container_width=True, hide_index=True)

    fig_paa = px.imshow(
        paa.values, x=paa.columns.tolist(), y=paa.index.tolist(),
        color_continuous_scale="RdYlGn", aspect="auto",
        labels=dict(color="PAA"),
    )
    fig_paa.update_layout(height=500)
    st.plotly_chart(fig_paa, use_container_width=True)

    # ── Leaderboard Race ────────────────────────────────────────────────────
    wc = c["wc"]

    st.subheader("Leaderboard Race")
    final_week = wc.iloc[-1]

    # derive weekly scores from cumulative
    weekly_scores = wc.diff()
    weekly_scores.iloc[0] = wc.iloc[0]

    st.markdown(
        "Double-Click **your** name in the legend to isolate your season and compare with others. Currently showing top 5.\n\n"
        "Shoutout to **b_hop** and his continued success over the course of the year, "
        "just missing the regular season crown and the Super Bowl crown. "
        "An incredible season overshadowed by razor-thin margins."
    )
    # show only the top 5 finishers by default
    top5_players = final_week.sort_values(ascending=False).head(5).index.tolist()
    fig_wc = go.Figure()
    for p in PLAYERS:
        if p in wc.columns:
            visible = True if p in top5_players else "legendonly"
            fig_wc.add_trace(go.Scatter(x=wc.index, y=wc[p], mode="lines+markers", name=p, visible=visible))
    fig_wc.update_layout(
        xaxis_title="Week", yaxis_title="Cumulative Correct Picks",
        height=500, margin=dict(t=10),
    )
    st.plotly_chart(fig_wc, use_container_width=True)

    # ── Rolling 3-Week Average ────────────────────────────────────────────
    st.subheader("Rolling 3-Week Average")
    st.markdown(
        "Who was hot and who was cold at any given point? "
        "This smooths out the week-to-week noise and shows momentum."
    )
    rolling_avg = weekly_scores.rolling(3, min_periods=1).mean()
    fig_rolling = go.Figure()
    for p in PLAYERS:
        if p in rolling_avg.columns:
            visible = True if p in top5_players else "legendonly"
            fig_rolling.add_trace(go.Scatter(
                x=rolling_avg.index, y=rolling_avg[p], mode="lines", name=p, visible=visible,
            ))
    fig_rolling.update_layout(
        xaxis_title="Week", yaxis_title="Avg Correct Picks (3-Week Rolling)",
        height=500, margin=dict(t=10),
    )
    st.plotly_chart(fig_rolling, use_container_width=True)

    # ── Hot & Cold Streaks ──────────────────────────────────────────────────
    streaks = c["streaks"]
    streaks = streaks[streaks["player"] != "Ripw1124"].reset_index(drop=True)

    prose_col, chart_col = st.columns([1, 2])
    with prose_col:
        st.subheader("Hot & Cold Streaks")
        best_row = streaks.loc[streaks["best_correct"].idxmax()]
        st.markdown(
            f"Best and worst 3-week windows for each manager.\n\n"
            f"**Exciting Whites** and **ADon** had historic hot streaks that "
            f"continued throughout the season, culminating in that tied regular season finish. "
            f"Meanwhile, **Yianni's** hot streak came in the playoffs, which is how he's "
            f"tied for the top heading into the Super Bowl.\n\n"
            f"**{best_row['player']}** had the single hottest stretch: "
            f"**{int(best_row['best_correct'])} correct** picks across "
            f"Weeks {best_row['best_weeks']}.\n\n"
            f"**MC$**, **Willheser**, and **Maye Magic** had a strong late-season surge, "
            f"but despite the momentum they all finished in the middle of the pack."
        )
    with chart_col:
        st.dataframe(streaks, use_container_width=True, hide_index=True)

    # ── Consensus & Contrarian ──────────────────────────────────────────────
    majority_df, contrarian_df = c["consensus"]
    contrarian_df = contrarian_df[contrarian_df["player"] != "Vegas"].reset_index(drop=True)

    prose_col, chart_col = st.columns([1, 2])
    with prose_col:
        st.subheader("Consensus & Contrarian")
        high_consensus = majority_df[majority_df["consensus_pct"] > 0.85]
        if len(high_consensus) > 0:
            rate = high_consensus["majority_correct"].mean()
            st.markdown(
                f"When >85% of the group agrees on a pick, the majority is right "
                f"**{rate:.0%}** of the time ({len(high_consensus)} games).\n\n"
            )
        top_contrarian = contrarian_df.iloc[0]
        st.markdown(
            f"It feels like beating a dead horse but **P-Otys** is the contrarian king, going "
            f"against the majority **{top_contrarian['contrarian_rate']:.0%}** of the time.\n\n"
            f"The bar color shows how often the contrarian pick actually won: "
            f"green = it pays off, red = maybe stop doing that."
        )
    with chart_col:
        fig_con = px.bar(
            contrarian_df.head(13), y="player", x="contrarian_rate", orientation="h",
            color="contrarian_win_rate",
            color_continuous_scale="RdYlGn",
            labels={"contrarian_rate": "Contrarian Rate", "contrarian_win_rate": "Win Rate When Contrarian", "player": ""},
        )
        fig_con.update_layout(height=450, yaxis=dict(autorange="reversed"), margin=dict(t=10))
        fig_con.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig_con, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: ML PICK PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Pick Predictor")
    st.caption(
        "Pick your name, generate a game, and see if the ML model can predict your pick."
    )

    if "predictor_total" not in st.session_state:
        st.session_state.predictor_total = 0
        st.session_state.predictor_correct = 0

    # ── Controls row (above the split) ────────────────────────────────────
    ctrl1, ctrl2 = st.columns([1, 1])
    with ctrl1:
        selected_player = st.selectbox("Who are you?", sorted(PLAYERS))
    with ctrl2:
        if selected_player in models:
            st.markdown("")  # spacer to align with selectbox label
            generate = st.button("Generate New Game", use_container_width=True)
        else:
            generate = False

    if selected_player not in models:
        st.warning(f"Not enough data to build a model for {selected_player}.")
    else:
        if "scenario" not in st.session_state or generate:
            away = random.choice(ALL_TEAMS)
            home = random.choice([t for t in ALL_TEAMS if t != away])
            spread_val = round(random.choice(np.arange(-10, 10.5, 0.5)), 1)
            week_val = random.randint(1, 18)
            is_indoor = random.random() < 0.3
            wind_val = 0 if is_indoor else random.randint(0, 25)
            rain = not is_indoor and random.random() < 0.15
            cross = conf_map.get(home, "NFC") != conf_map.get(away, "NFC")

            st.session_state.scenario = {
                "home": home, "away": away, "spread": spread_val, "week": week_val,
                "indoor": is_indoor, "wind": wind_val,
                "rain_snow": rain, "cross_conf": cross,
            }
            st.session_state.pop("user_pick", None)

        if "scenario" in st.session_state:
            s = st.session_state.scenario

            # compute model confidence up front
            model = models[selected_player]
            prob_home = predict_pick(
                model, feature_names,
                home_team=s["home"], away_team=s["away"], spread=s["spread"],
                week=s["week"], indoor=s["indoor"], temp=65,
                wind=s["wind"], rain_snow=s["rain_snow"], cross_conf=s["cross_conf"],
            )
            prob_away = 1 - prob_home
            model_pick = s["home"] if prob_home >= 0.5 else s["away"]
            model_conf = max(prob_home, prob_away)

            st.divider()

            # ── Two-panel layout with divider ─────────────────────────
            left_col, divider_col, right_col = st.columns([10, 1, 10])

            with left_col:
                spread_display = f"{s['home']} {s['spread']:+.1f}" if s["spread"] < 0 else f"{s['away']} {-s['spread']:+.1f}"
                weather_str = "Indoor" if s["indoor"] else f"{s['wind']} mph wind" + (", rain/snow" if s["rain_snow"] else "")
                conf_str = "Cross-conference" if s["cross_conf"] else "Same conference"

                st.markdown(f"### Week {s['week']}: {s['away']} @ {s['home']}")
                st.caption(f"Model confidence for this game: **{model_conf:.0%}**")
                st.markdown(f"**Spread:** {spread_display}")
                st.markdown(f"**Weather:** {weather_str} | {conf_str}")

                pick_l, pick_r = st.columns(2)
                picked = None
                with pick_l:
                    if st.button(f"Pick {s['away']} (Away)", use_container_width=True, type="secondary"):
                        picked = s["away"]
                with pick_r:
                    if st.button(f"Pick {s['home']} (Home)", use_container_width=True, type="secondary"):
                        picked = s["home"]

                if picked is not None:
                    st.session_state.predictor_total += 1
                    if picked == model_pick:
                        st.session_state.predictor_correct += 1
                    # generate new game and rerun
                    st.session_state.pop("scenario", None)
                    st.rerun()

            with divider_col:
                st.markdown(
                    "<div style='border-left: 2px solid #ccc; height: 100%; min-height: 400px;'></div>",
                    unsafe_allow_html=True,
                )

            with right_col:
                total = st.session_state.predictor_total
                correct = st.session_state.predictor_correct
                if total > 0:
                    pct = f" ({correct/total:.0%})"
                    st.success(f"Model score: {correct}/{total} correct{pct}")

                st.markdown("### Make your pick")
                st.markdown("Pick a team on the left to see if the model predicted your choice.")

                fig_prob = go.Figure(go.Bar(
                    x=[prob_away, prob_home],
                    y=["Away", "Home"],
                    orientation="h",
                    marker_color=["#EF553B", "#636EFA"],
                    text=[f"{prob_away:.0%}", f"{prob_home:.0%}"],
                    textposition="auto",
                ))
                fig_prob.update_layout(
                    xaxis_title="Probability", yaxis_title="",
                    height=180, margin=dict(t=10, b=20),
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                # ── Why the model thinks this ─────────────────────
                st.markdown("**Why the model thinks this:**")
                feat_row = {name: 0 for name in feature_names}
                feat_row["spread"] = s["spread"]
                feat_row["spread_abs"] = abs(s["spread"])
                feat_row["week"] = s["week"]
                feat_row[f"home_{s['home']}"] = 1
                feat_row[f"away_{s['away']}"] = 1
                feat_row["cross_conference"] = int(s["cross_conf"])
                feat_row["indoor"] = int(s["indoor"])
                feat_row["temp"] = 65
                feat_row["wind"] = s["wind"]
                feat_row["rain_snow"] = int(s["rain_snow"])

                # get PAA for this player's team preferences
                player_paa = c["paa"]
                player_paa_row = {}
                if selected_player in player_paa.index:
                    player_paa_row = player_paa.loc[selected_player].to_dict()

                coefs = model.coef_[0]
                contributions = []
                for i, fname in enumerate(feature_names):
                    val = feat_row[fname]
                    contrib = coefs[i] * val
                    if abs(contrib) > 0.01:
                        if fname.startswith("home_"):
                            team_id = fname[5:]
                            paa_val = player_paa_row.get(team_id, 0)
                            paa_note = f" (PAA: {paa_val:+.1f})" if abs(paa_val) >= 1 else ""
                            label = f"{team_id} at home{paa_note}"
                        elif fname.startswith("away_"):
                            team_id = fname[5:]
                            paa_val = player_paa_row.get(team_id, 0)
                            paa_note = f" (PAA: {paa_val:+.1f})" if abs(paa_val) >= 1 else ""
                            label = f"{team_id} on the road{paa_note}"
                        elif fname == "spread":
                            label = f"Spread ({val:+.1f})"
                        elif fname == "spread_abs":
                            label = f"Spread size ({val:.1f} pts)"
                        elif fname == "temp":
                            continue
                        elif fname == "wind":
                            label = f"Wind ({int(val)} mph)"
                        elif fname == "indoor":
                            label = "Indoor game" if val else "Outdoor game"
                        elif fname == "rain_snow":
                            label = "Rain/snow" if val else "Clear weather"
                        elif fname == "cross_conference":
                            label = "Cross-conference" if val else "Same conference"
                        elif fname == "week":
                            label = f"Week {int(val)}"
                        else:
                            label = fname
                        contributions.append((label, contrib))

                contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                top_factors = contributions[:4]
                reasons = []
                for label, contrib in top_factors:
                    direction = "toward home" if contrib > 0 else "toward away"
                    reasons.append(f"- {label} pushes {direction}")
                st.markdown("\n".join(reasons))
