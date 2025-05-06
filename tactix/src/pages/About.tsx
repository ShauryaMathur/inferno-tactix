import React from "react";
import styles from "./about.module.scss";
import PPOImage from "../assets/ppo_policy_thumb.png";

/**
 * About page (CSS-Modules version)
 * Drop this component into your router at "/about".
 * It relies only on about.module.scss – no Tailwind / shadcn required.
 */
export default function About() {
  return (
    <div className={styles.about}>
      {/* Hero */}
      <h1 className={styles.title}>Inferno <span style={{color: "#ffd54f"}}>Tactics</span></h1>
      <p className={styles.subtitle}>
        Proactive wildfire prediction and intelligent suppression driven by Deep&nbsp;Learning&nbsp;+&nbsp;Reinforcement&nbsp;Learning.
      </p>

      {/* Card grid */}
      <div className={styles.grid}>
        {/* Wildfire problem card */}
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>Wildfires—Why it matters</h2>
          <p className={styles.cardBody}>
            U.S. wildfires now scorch <strong>10&nbsp;million&nbsp;acres</strong> annually, with suppression costs exceeding
            <strong> $4&nbsp;billion</strong>. Beyond dollars, flames threaten communities, ecosystems, and air quality.
            Reactive approaches can’t keep pace; we need prediction and intelligent response.
          </p>
        </section>

        {/* Inferno (DL) card */}
        <section className={`${styles.card} ${styles.cardHorizontal}`}>
          {/* <img src="/assets/cnn_bilstm_thumb.png" alt="CNN–BiLSTM" /> */}
          <div>
            <h2 className={styles.cardTitle}>Inferno – Deep-Learning Forecasts</h2>
            <p className={styles.cardBody}>
              A CNN-BiLSTM model ingests 75-day GRIDMET weather histories for every ignition point
              and hits <strong>84.79&nbsp;% accuracy</strong> on unseen test regions—flagging high-risk periods up to two weeks ahead.
            </p>
          </div>
        </section>

        {/* Tactics (RL) card */}
        <section className={`${styles.card} ${styles.cardHorizontal}`}>
          {/* <img src="/assets/ppo_policy_thumb.png" alt="PPO helitack policy" /> */}
          <img src={PPOImage} alt="PPO helitack policy" />

          <div>
            <h2 className={styles.cardTitle}>Tactics – RL Helitack Agent</h2>
            <p className={styles.cardBody}>
              A Three.js environment with a physics-based fire engine feeds stacked fire-growth frames to a
              PPO agent. The learned helitack policy circles hotspots and cuts spread, reducing burn area by
              <strong> ~50&nbsp;%</strong> versus scripted baselines.
            </p>
          </div>
        </section>

        {/* One-pager report */}
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>From bytes to briefing</h2>
          <p className={styles.cardBody}>
            With one click, field commanders receive a concise PDF: timestamped risk score, predicted burn curve,
            and GPS coordinates for optimal helitack deployment under the RL policy.
          </p>
          <a href="/sample-report.pdf" className={styles.button}>
            View sample report <span aria-hidden="true">→</span>
          </a>
        </section>
      </div>
    </div>
  );
}