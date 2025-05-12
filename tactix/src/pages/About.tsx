import React from "react";
import { Flame, AlertTriangle, Brain, FileText } from "lucide-react";
import styles from "./about.module.scss";
import PPOImage from "../assets/ppo_policy_thumb.png";
import infernix from "../assets/infernix.png";

export default function About() {
  return (
    <div className={styles.container}>
      <div className={styles.hero}>
        <h1 className={styles.title}>
          Inferno <span className={styles.highlight}>Tactics</span>
        </h1>
        <p className={styles.subtitle}>
          Proactive wildfire prediction and intelligent suppression driven by
          <span className={styles.emphasis}> Deep Learning + Reinforcement Learning.</span>
        </p>
      </div>

      <div className={styles.grid}>
        <section className={`${styles.card} ${styles.problemCard}`}>
          <div className={styles.cardHeader}>
            <AlertTriangle className={styles.icon} />
            <h2 className={styles.cardTitle}>Wildfires—Why it matters</h2>
          </div>
          <div className={styles.cardContent}>
            <div className={styles.imageContainer}>
              <img 
                src="https://images.pexels.com/photos/14840720/pexels-photo-14840720.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
                alt="Intense wildfire burning through forest" 
                className={styles.cardImage} 
              />
            </div>
            <p className={styles.cardBody}>
              U.S. wildfires now scorch <strong>10&nbsp;million&nbsp;acres</strong> annually, with suppression costs exceeding
              <strong> $4&nbsp;billion</strong>. Beyond dollars, flames threaten communities, ecosystems, and air quality.
              Reactive approaches can't keep pace; we need prediction and intelligent response.
            </p>
          </div>
        </section>

        <section className={`${styles.card} ${styles.infernoCard}`}>
          <div className={styles.cardHeader}>
            <Flame className={styles.icon} />
            <h2 className={styles.cardTitle}>Inferno – Deep-Learning Forecasts</h2>
          </div>
          <div className={styles.cardContent}>
          <div className={styles.imageContainer}>
              <img 
                src={infernix}
                alt="AI Infernix Model Visualization" 
                className={styles.cardImage}
              />
              
            </div>
            <p className={styles.cardBody}>
              A CNN-BiLSTM model ingests 75-day GRIDMET weather histories for every ignition point
              and hits <strong>84.79&nbsp;% accuracy</strong> on unseen test regions—flagging high-risk periods up to two weeks ahead.
            </p>
          </div>
        </section>

        <section className={`${styles.card} ${styles.tacticsCard}`}>
          <div className={styles.cardHeader}>
            <Brain className={styles.icon} />
            <h2 className={styles.cardTitle}>Tactics – RL Helitack Agent</h2>
          </div>
          <div className={styles.cardContent}>
            <div className={styles.imageContainer}>
              <img 
                src={PPOImage} 
                alt="AI visualization representing the helitack policy" 
                className={styles.cardImage} 
              />
            </div>
            <p className={styles.cardBody}>
              A Three.js environment with a physics-based fire engine feeds stacked fire-growth frames to a
              PPO agent. The learned helitack policy circles hotspots and cuts spread, reducing burn area by
              <strong> ~50&nbsp;%</strong> versus scripted baselines.
            </p>
          </div>
        </section>

        <section className={`${styles.card} ${styles.reportCard}`}>
          <div className={styles.cardHeader}>
            <FileText className={styles.icon} />
            <h2 className={styles.cardTitle}>From bytes to briefing</h2>
          </div>
          <div className={styles.cardContent}>
            <div className={styles.imageContainer}>
              <img 
                src="https://images.pexels.com/photos/9835606/pexels-photo-9835606.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2" 
                alt="Digital report visualization" 
                className={styles.cardImage} 
              />
            </div>
            <p className={styles.cardBody}>
              With one click, field commanders receive a concise PDF: timestamped risk score, predicted burn curve,
              and GPS coordinates for optimal <strong>helitack</strong> deployment under the RL policy.
            </p>
          </div>
          <a href="https://buffalo.app.box.com/file/1854274825450?s=jpljoklsw6y68xeq9bdxtvc7jy6b12kg" className={styles.button}>
            View sample report <span aria-hidden="true">→</span>
          </a>
        </section>
      </div>
    </div>
  );
}