import React from 'react';
import styles from './home.module.scss'

const Home: React.FC = () => {
  // const backgroundUrl = './bg_home.png';  // ← update this path

  return (
    <div
      className={styles.home}
    >
      <h1 className={styles.title}>
        Inferno Tactics
      </h1>
    </div>
  );
};

export default Home;