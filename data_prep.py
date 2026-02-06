"""
Precompute all stats for the RYP dashboard from scores_and_picks.csv + picks.csv.
Imported by app.py — all functions return DataFrames or dicts.
"""
import pandas as pd
import numpy as np

PLAYERS = [
    "ADon", "Exciting Whites", "Kevin", "MC$", "Maye Magic", "P-Otys",
    "Ripw1124", "Vegas", "Willheser", "Yianni", "b_hop", "derelicious", "mrmcwinnerson",
]


def load_data():
    sap = pd.read_csv("scores_and_picks.csv")
    picks = pd.read_csv("picks.csv")
    teams_df = pd.read_csv("nfl_teams (1).csv")
    return sap, picks, teams_df


# ── Tab 1: Team Performance ─────────────────────────────────────────────────

def team_ats_record(sap):
    """ATS record for every team (as home + away combined)."""
    rows = []
    all_teams = set(sap["team_home"].dropna()) | set(sap["team_away"].dropna())
    for team in sorted(all_teams):
        home = sap[sap["team_home"] == team]
        away = sap[sap["team_away"] == team]
        covers_home = (home["ats_winner"] == team).sum()
        covers_away = (away["ats_winner"] == team).sum()
        games_home = len(home)
        games_away = len(away)
        total_covers = covers_home + covers_away
        total_games = games_home + games_away
        pushes_home = (home["ats_winner"] == "PUSH").sum()
        pushes_away = (away["ats_winner"] == "PUSH").sum()
        rows.append({
            "team": team,
            "ats_wins": total_covers,
            "ats_losses": total_games - total_covers - pushes_home - pushes_away,
            "ats_pushes": pushes_home + pushes_away,
            "ats_games": total_games,
            "ats_pct": total_covers / total_games if total_games else 0,
        })
    return pd.DataFrame(rows).sort_values("ats_pct", ascending=False).reset_index(drop=True)


def team_ml_record(sap):
    """Straight-up (moneyline) record for every team."""
    rows = []
    all_teams = set(sap["team_home"].dropna()) | set(sap["team_away"].dropna())
    for team in sorted(all_teams):
        home = sap[sap["team_home"] == team]
        away = sap[sap["team_away"] == team]
        wins_home = (home["score_home"] > home["score_away"]).sum()
        wins_away = (away["score_away"] > away["score_home"]).sum()
        total_wins = wins_home + wins_away
        total_games = len(home) + len(away)
        rows.append({
            "team": team,
            "ml_wins": total_wins,
            "ml_losses": total_games - total_wins,
            "ml_games": total_games,
            "ml_pct": total_wins / total_games if total_games else 0,
        })
    return pd.DataFrame(rows).sort_values("ml_pct", ascending=False).reset_index(drop=True)


def home_away_ats(sap):
    """ATS cover rate split by home vs away for each team."""
    rows = []
    all_teams = set(sap["team_home"].dropna()) | set(sap["team_away"].dropna())
    for team in sorted(all_teams):
        home = sap[sap["team_home"] == team]
        away = sap[sap["team_away"] == team]
        home_covers = (home["ats_winner"] == team).sum()
        away_covers = (away["ats_winner"] == team).sum()
        rows.append({
            "team": team,
            "home_cover_pct": home_covers / len(home) if len(home) else 0,
            "away_cover_pct": away_covers / len(away) if len(away) else 0,
            "home_games": len(home),
            "away_games": len(away),
        })
    return pd.DataFrame(rows)


def spread_impact(sap):
    """Favorite ATS cover rate bucketed by spread magnitude."""
    df = sap.dropna(subset=["platform_spread", "ats_winner"]).copy()
    df["spread_abs"] = df["platform_spread"].abs()
    # determine favorite: negative spread → home, positive → away
    df["favorite"] = np.where(df["platform_spread"] < 0, df["team_home"], df["team_away"])
    df["fav_covered"] = (df["ats_winner"] == df["favorite"]).astype(int)
    df["spread_bucket"] = pd.cut(df["spread_abs"], bins=[0, 1, 3, 5, 7, 10, 20], labels=["0.5-1", "1.5-3", "3.5-5", "5.5-7", "7.5-10", "10+"])
    return df.groupby("spread_bucket", observed=True).agg(
        fav_cover_rate=("fav_covered", "mean"),
        games=("fav_covered", "count"),
    ).reset_index()


def weekly_surprise(sap):
    """Per-week: what % of favorites covered (lower = more upsets)."""
    df = sap.dropna(subset=["platform_spread", "ats_winner"]).copy()
    df["favorite"] = np.where(df["platform_spread"] < 0, df["team_home"], df["team_away"])
    df["fav_covered"] = (df["ats_winner"] == df["favorite"]).astype(int)
    return df.groupby("week").agg(
        fav_cover_rate=("fav_covered", "mean"),
        games=("fav_covered", "count"),
    ).reset_index().sort_values("week")


# ── Tab 2: Bias & Patterns ──────────────────────────────────────────────────

def most_picked_teams(picks):
    """Total picks per team across all players."""
    return picks.groupby("pick").size().reset_index(name="total_picks").sort_values("total_picks", ascending=False)


def player_fav_underdog_rate(picks, sap):
    """How often each player picks the favorite vs underdog."""
    # merge to get which team is the favorite
    merged = picks.merge(
        sap[["week", "game", "team_home", "team_away", "platform_spread"]].drop_duplicates(),
        on=["week", "game"], how="inner",
    )
    merged["favorite"] = np.where(merged["platform_spread"] < 0, merged["team_home"], merged["team_away"])
    merged["picked_fav"] = (merged["pick"] == merged["favorite"]).astype(int)
    return merged.groupby("player").agg(
        fav_rate=("picked_fav", "mean"),
        total=("picked_fav", "count"),
    ).reset_index().sort_values("fav_rate", ascending=False)


def paa_heatmap(picks):
    """Picks Above Average: player × team matrix of pick counts minus league average."""
    ct = picks.groupby(["player", "pick"]).size().unstack(fill_value=0)
    avg = ct.mean(axis=0)
    return ct.sub(avg)


def weekly_cumulative(sap):
    """Running total of correct ATS picks per player over 18 weeks."""
    weekly = sap.groupby("week")[PLAYERS].sum()
    return weekly.cumsum()


def hot_cold_streaks(sap):
    """Best and worst weekly streaks per player."""
    weekly = sap.groupby("week")[PLAYERS].sum().sort_index()
    results = {}
    for p in PLAYERS:
        if p not in weekly.columns:
            continue
        scores = weekly[p].values
        weeks = weekly.index.values
        # find best and worst consecutive-week windows (size 3)
        best_sum, best_start = -1, 0
        worst_sum, worst_start = 999, 0
        for i in range(len(scores) - 2):
            s = scores[i] + scores[i + 1] + scores[i + 2]
            if s > best_sum:
                best_sum, best_start = s, i
            if s < worst_sum:
                worst_sum, worst_start = s, i
        results[p] = {
            "best_weeks": f"{weeks[best_start]}-{weeks[best_start+2]}",
            "best_correct": int(best_sum),
            "worst_weeks": f"{weeks[worst_start]}-{weeks[worst_start+2]}",
            "worst_correct": int(worst_sum),
        }
    return pd.DataFrame(results).T.reset_index().rename(columns={"index": "player"})


def consensus_contrarian(picks, sap):
    """For each game, what did the majority pick? How often was the majority right?
    Who goes contrarian most?"""
    # majority pick per game
    game_picks = picks.groupby(["week", "game", "pick"]).size().reset_index(name="n")
    idx = game_picks.groupby(["week", "game"])["n"].idxmax()
    majority = game_picks.loc[idx, ["week", "game", "pick", "n"]].rename(columns={"pick": "majority_pick", "n": "majority_count"})

    # total pickers per game
    total_per_game = picks.groupby(["week", "game"]).size().reset_index(name="total_pickers")
    majority = majority.merge(total_per_game, on=["week", "game"])
    majority["consensus_pct"] = majority["majority_count"] / majority["total_pickers"]

    # was majority right?
    majority = majority.merge(sap[["week", "game", "ats_winner"]], on=["week", "game"], how="inner")
    majority["majority_correct"] = (majority["majority_pick"] == majority["ats_winner"]).astype(int)

    # contrarian rate per player
    player_picks = picks.merge(majority[["week", "game", "majority_pick"]], on=["week", "game"])
    player_picks["is_contrarian"] = (player_picks["pick"] != player_picks["majority_pick"]).astype(int)

    contrarian = player_picks.groupby("player").agg(
        contrarian_rate=("is_contrarian", "mean"),
        contrarian_count=("is_contrarian", "sum"),
        total=("is_contrarian", "count"),
    ).reset_index().sort_values("contrarian_rate", ascending=False)

    # contrarian success rate
    player_picks = player_picks.merge(sap[["week", "game", "ats_winner"]], on=["week", "game"])
    player_picks["correct"] = (player_picks["pick"] == player_picks["ats_winner"]).astype(int)
    contrarian_only = player_picks[player_picks["is_contrarian"] == 1]
    contrarian_success = contrarian_only.groupby("player")["correct"].mean().reset_index(name="contrarian_win_rate")
    contrarian = contrarian.merge(contrarian_success, on="player", how="left")

    return majority, contrarian


def herd_mentality(picks, sap):
    """How often each player agrees with the majority pick."""
    game_picks = picks.groupby(["week", "game", "pick"]).size().reset_index(name="n")
    idx = game_picks.groupby(["week", "game"])["n"].idxmax()
    majority = game_picks.loc[idx, ["week", "game", "pick"]].rename(columns={"pick": "majority_pick"})

    merged = picks.merge(majority, on=["week", "game"])
    merged["with_herd"] = (merged["pick"] == merged["majority_pick"]).astype(int)
    return merged.groupby("player")["with_herd"].mean().reset_index(name="herd_rate").sort_values("herd_rate", ascending=False)
