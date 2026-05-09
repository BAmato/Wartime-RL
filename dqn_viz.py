"""
DQN Training Dashboard
======================
Usage:
    python dqn_viz.py dqn_training_log.csv
    python dqn_viz.py dqn_training_log.csv --window 20 --xaxis episode

Arguments:
    csv_path        Path to the training log CSV
    --window N      Smoothing window size (default: 20)
    --xaxis         X-axis units: 'episode' or 'step' (default: step)
    --save PATH     Save figure to file instead of showing (e.g. out.png)

Expected CSV columns:
    episode, global_step, outcome, ep_steps, reward,
    agent_terr, enemy_terr, epsilon, curriculum_level, mean_loss
    Optional: turns (full agent turns per episode)
"""

import argparse
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── colour palette ────────────────────────────────────────────────────────────
DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d27"
GRID_COL  = "#2a2d3a"
TEXT_COL  = "#e0e2ef"
MUTED_COL = "#5a5e7a"

C_WIN     = "#4ecdc4"   # teal
C_LOSS    = "#ff6b6b"   # red
C_TIMEOUT = "#ffd166"   # yellow
C_SMOOTH  = "#ffffff"
C_LOSS_CURVE = "#a78bfa"  # purple
C_STEPS   = "#74b9ff"   # blue
C_TURNS   = "#a8e6cf"   # mint
C_AGENT   = "#4ecdc4"
C_ENEMY   = "#ff6b6b"
C_EPSILON = "#fdcb6e"

OUTCOME_COLORS = {"win": C_WIN, "loss": C_LOSS, "timeout": C_TIMEOUT}


# ── helpers ───────────────────────────────────────────────────────────────────

def rolling(series: pd.Series, w: int, min_periods: int = 1) -> pd.Series:
    return series.rolling(w, min_periods=min_periods).mean()


def outcome_rates(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """Rolling win/loss/timeout rates as fractions (sum = 1)."""
    for o in ("win", "loss", "timeout"):
        df[f"_is_{o}"] = (df["outcome"] == o).astype(float)
    rates = pd.DataFrame(index=df.index)
    for o in ("win", "loss", "timeout"):
        rates[o] = rolling(df[f"_is_{o}"], w)
    return rates


def style_ax(ax, title: str, ylabel: str = "", xlabel: str = ""):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=9, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, color=MUTED_COL, fontsize=8)
    ax.set_xlabel(xlabel, color=MUTED_COL, fontsize=8)
    ax.tick_params(colors=MUTED_COL, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_xlim(left=0)


def shade_curriculum(ax, df: pd.DataFrame, x: pd.Series):
    """Light vertical bands whenever curriculum_level changes."""
    if df["curriculum_level"].nunique() <= 1:
        return
    prev_level = df["curriculum_level"].iloc[0]
    band_start = x.iloc[0]
    colors = plt.cm.tab10.colors
    for i, (xi, lvl) in enumerate(zip(x, df["curriculum_level"])):
        if lvl != prev_level:
            ax.axvspan(band_start, xi,
                       color=colors[int(prev_level) % 10], alpha=0.06,
                       linewidth=0)
            band_start = xi
            prev_level = lvl
    ax.axvspan(band_start, x.iloc[-1],
               color=colors[int(prev_level) % 10], alpha=0.06, linewidth=0)


# ── main plot ─────────────────────────────────────────────────────────────────

def build_dashboard(df: pd.DataFrame, window: int, xaxis: str, save: str | None):

    x      = df["global_step"] if xaxis == "step" else df["episode"]
    xlabel = "Global Step" if xaxis == "step" else "Episode"

    # Prefer turns column for episode-length plot when available
    has_turns = "turns" in df.columns
    length_col = "turns" if has_turns else "ep_steps"
    length_label = "Turns" if has_turns else "Steps"

    rates  = outcome_rates(df, window)

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle(
        "DQN Training Dashboard",
        color=TEXT_COL, fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.06, right=0.97, top=0.93, bottom=0.07
    )

    ax_reward   = fig.add_subplot(gs[0, :2])   # row 0, cols 0-1 (wide)
    ax_winrate  = fig.add_subplot(gs[0, 2])    # row 0, col 2
    ax_loss     = fig.add_subplot(gs[1, 0])
    ax_steps    = fig.add_subplot(gs[1, 1])
    ax_terr     = fig.add_subplot(gs[1, 2])
    ax_epsilon  = fig.add_subplot(gs[2, 0])
    ax_rew_dist = fig.add_subplot(gs[2, 1])
    ax_outcome  = fig.add_subplot(gs[2, 2])

    # ── 1. Reward curve (primary diagnostic) ─────────────────────────────────
    for outcome, color in OUTCOME_COLORS.items():
        mask = df["outcome"] == outcome
        ax_reward.scatter(
            x[mask], df.loc[mask, "reward"],
            c=color, s=18, alpha=0.45, linewidths=0, label=outcome.capitalize()
        )

    sm_reward = rolling(df["reward"], window)
    std_reward = df["reward"].rolling(window, min_periods=1).std().fillna(0)
    ax_reward.plot(x, sm_reward, color=C_SMOOTH, linewidth=1.8, zorder=5, label=f"Smoothed (w={window})")
    ax_reward.fill_between(x, sm_reward - std_reward, sm_reward + std_reward,
                           color=C_SMOOTH, alpha=0.10, zorder=4)
    ax_reward.axhline(0, color=MUTED_COL, linewidth=0.7, linestyle=":")
    shade_curriculum(ax_reward, df, x)
    style_ax(ax_reward, "Episode Reward", ylabel="Reward", xlabel=xlabel)
    leg = ax_reward.legend(
        loc="upper left", fontsize=7, framealpha=0.3,
        facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL
    )

    # ── 2. Win / Loss / Timeout rolling rates ─────────────────────────────────
    ax_winrate.stackplot(
        x,
        rates["win"], rates["timeout"], rates["loss"],
        colors=[C_WIN, C_TIMEOUT, C_LOSS],
        alpha=0.80
    )
    ax_winrate.set_ylim(0, 1)
    patches = [mpatches.Patch(color=c, label=l)
               for c, l in [(C_WIN, "Win"), (C_TIMEOUT, "Timeout"), (C_LOSS, "Loss")]]
    ax_winrate.legend(
        handles=patches, loc="lower right", fontsize=6.5,
        framealpha=0.3, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL
    )
    style_ax(ax_winrate, f"Outcome Rates (rolling {window})", ylabel="Rate", xlabel=xlabel)

    # ── 3. Training loss ──────────────────────────────────────────────────────
    ax_loss.scatter(x, df["mean_loss"], c=C_LOSS_CURVE, s=12, alpha=0.35, linewidths=0)
    ax_loss.plot(x, rolling(df["mean_loss"], window), color=C_LOSS_CURVE, linewidth=1.6)
    style_ax(ax_loss, "Mean TD Loss", ylabel="Loss", xlabel=xlabel)

    # ── 4. Episode length (turns preferred, steps fallback) ───────────────────
    for outcome, color in OUTCOME_COLORS.items():
        mask = df["outcome"] == outcome
        ax_steps.scatter(x[mask], df.loc[mask, length_col],
                         c=color, s=14, alpha=0.5, linewidths=0)
    ax_steps.plot(x, rolling(df[length_col], window), color=C_TURNS if has_turns else C_STEPS,
                  linewidth=1.5)
    ax_steps.axhline(df[length_col].max(), color=C_TIMEOUT, linewidth=0.7,
                     linestyle=":", alpha=0.6, label=f"Max {length_label.lower()}")
    # Overlay raw steps as faint secondary line when showing turns
    if has_turns:
        ax_steps.plot(x, rolling(df["ep_steps"], window),
                      color=C_STEPS, linewidth=0.8, alpha=0.4, linestyle="--",
                      label="Steps (raw)")
        ax_steps.legend(loc="upper right", fontsize=6.5, framealpha=0.3,
                        facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    style_ax(ax_steps, f"Episode Length ({length_label})", ylabel=length_label, xlabel=xlabel)

    # ── 5. Territory control ──────────────────────────────────────────────────
    ax_terr.plot(x, rolling(df["agent_terr"], window),
                 color=C_AGENT, linewidth=1.5, label="Agent")
    ax_terr.plot(x, rolling(df["enemy_terr"], window),
                 color=C_ENEMY, linewidth=1.5, label="Enemy")
    ax_terr.fill_between(x,
                         rolling(df["agent_terr"], window),
                         rolling(df["enemy_terr"], window),
                         where=(rolling(df["agent_terr"], window) >=
                                rolling(df["enemy_terr"], window)),
                         color=C_AGENT, alpha=0.15, label="_")
    ax_terr.fill_between(x,
                         rolling(df["agent_terr"], window),
                         rolling(df["enemy_terr"], window),
                         where=(rolling(df["agent_terr"], window) <
                                rolling(df["enemy_terr"], window)),
                         color=C_ENEMY, alpha=0.15, label="_")
    ax_terr.legend(loc="upper left", fontsize=7, framealpha=0.3,
                   facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    style_ax(ax_terr, "Territory Control (smoothed)", ylabel="Tiles", xlabel=xlabel)

    # ── 6. Epsilon (exploration) schedule ─────────────────────────────────────
    ax_epsilon.plot(x, df["epsilon"], color=C_EPSILON, linewidth=1.5)
    ax_epsilon.set_ylim(0, 1.05)
    ax_epsilon.annotate(
        f"ε = {df['epsilon'].iloc[-1]:.4f}",
        xy=(x.iloc[-1], df["epsilon"].iloc[-1]),
        xytext=(-40, 10), textcoords="offset points",
        color=C_EPSILON, fontsize=7,
        arrowprops=dict(arrowstyle="->", color=C_EPSILON, lw=0.8)
    )
    style_ax(ax_epsilon, "Epsilon (Exploration)", ylabel="ε", xlabel=xlabel)

    # ── 7. Reward distribution by outcome (violin / kde fallback) ─────────────
    try:
        from scipy.stats import gaussian_kde
        for outcome, color in OUTCOME_COLORS.items():
            vals = df.loc[df["outcome"] == outcome, "reward"].values
            if len(vals) < 3:
                continue
            kde = gaussian_kde(vals, bw_method="scott")
            y_range = np.linspace(vals.min() - 0.3, vals.max() + 0.3, 200)
            density = kde(y_range)
            density /= density.max()
            offset = {"win": 1, "timeout": 2, "loss": 3}[outcome]
            ax_rew_dist.fill_betweenx(y_range, offset - density * 0.4,
                                      offset + density * 0.4,
                                      color=color, alpha=0.6)
            ax_rew_dist.scatter([offset] * len(vals), vals,
                                c=color, s=8, alpha=0.4, zorder=5)
        ax_rew_dist.set_xticks([1, 2, 3])
        ax_rew_dist.set_xticklabels(["Win", "Timeout", "Loss"],
                                    color=MUTED_COL, fontsize=7)
        ax_rew_dist.set_xlim(0.4, 3.6)
    except ImportError:
        data_by_outcome = [
            df.loc[df["outcome"] == o, "reward"].values
            for o in ("win", "timeout", "loss")
        ]
        bp = ax_rew_dist.boxplot(
            data_by_outcome, patch_artist=True,
            medianprops=dict(color="white", linewidth=1.5)
        )
        for patch, color in zip(bp["boxes"], [C_WIN, C_TIMEOUT, C_LOSS]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_rew_dist.set_xticklabels(["Win", "Timeout", "Loss"],
                                    color=MUTED_COL, fontsize=7)

    ax_rew_dist.axhline(0, color=MUTED_COL, linewidth=0.7, linestyle=":")
    style_ax(ax_rew_dist, "Reward Distribution by Outcome", ylabel="Reward")

    # ── 8. Cumulative outcome counts ──────────────────────────────────────────
    for outcome, color in OUTCOME_COLORS.items():
        cumcount = (df["outcome"] == outcome).cumsum()
        ax_outcome.plot(x, cumcount, color=color, linewidth=1.5,
                        label=outcome.capitalize())
    ax_outcome.legend(loc="upper left", fontsize=7, framealpha=0.3,
                      facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    style_ax(ax_outcome, "Cumulative Outcome Counts", ylabel="Count", xlabel=xlabel)

    # ── stats annotation ──────────────────────────────────────────────────────
    n_ep = len(df)
    wr   = (df["outcome"] == "win").mean() * 100
    lr   = (df["outcome"] == "loss").mean() * 100
    tr   = (df["outcome"] == "timeout").mean() * 100
    last_eps = df["epsilon"].iloc[-1]
    avg_r = df["reward"].mean()
    info  = (f"Episodes: {n_ep}  |  Steps: {df['global_step'].iloc[-1]:,}  |  "
             f"W: {wr:.0f}%  L: {lr:.0f}%  T: {tr:.0f}%  |  "
             f"Avg R: {avg_r:.3f}  |  ε: {last_eps:.4f}  |  "
             f"Smooth window: {window}")
    fig.text(0.5, 0.005, info, ha="center", va="bottom",
             color=MUTED_COL, fontsize=7.5)

    # ── save or show ──────────────────────────────────────────────────────────
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Saved to {save}")
    else:
        plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DQN Training Dashboard")
    parser.add_argument("csv_path", help="Path to training log CSV")
    parser.add_argument("--window", type=int, default=20,
                        help="Rolling average window (default: 20)")
    parser.add_argument("--xaxis", choices=["step", "episode"], default="step",
                        help="X-axis: global step count or episode number (default: step)")
    parser.add_argument("--save", default=None,
                        help="Output file path (e.g. dashboard.png). Omit to display interactively.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, comment="#")

    required = {"episode", "global_step", "outcome", "ep_steps", "reward",
                "agent_terr", "enemy_terr", "epsilon", "curriculum_level", "mean_loss"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df["outcome"] = df["outcome"].str.strip().str.lower()

    # Clamp window to something reasonable
    window = min(args.window, max(1, len(df) // 3))
    if window != args.window:
        print(f"Note: window clamped to {window} (only {len(df)} episodes)")

    build_dashboard(df, window=window, xaxis=args.xaxis, save=args.save)


if __name__ == "__main__":
    main()
