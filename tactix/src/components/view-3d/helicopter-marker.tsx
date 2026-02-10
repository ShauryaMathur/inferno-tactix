import React, { RefObject } from "react";
import { observer } from "mobx-react";
import { useStores } from "../../use-stores";
import { Marker } from "./marker";
import * as THREE from "three";

interface IProps {
  dragPlane: RefObject<THREE.Mesh>
}

// Simple helicopter icon - using a circular indicator
const createHelicopterIcon = (): HTMLCanvasElement => {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  
  if (ctx) {
    // Draw outer circle (background)
    ctx.fillStyle = '#000000';
    ctx.beginPath();
    ctx.arc(32, 32, 30, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw inner circle (helicopter body)
    ctx.fillStyle = '#FF4444';
    ctx.beginPath();
    ctx.arc(32, 32, 20, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw rotor lines
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(12, 32);
    ctx.lineTo(52, 32);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(32, 12);
    ctx.lineTo(32, 52);
    ctx.stroke();
  }
  
  return canvas;
};

const helicopterIcon = createHelicopterIcon();

export const HelicopterMarker: React.FC<IProps> = observer(function WrappedComponent({ dragPlane }) {
  const { simulation } = useStores();
  
  if (!simulation.dataReady || !simulation.helicopterPosition) {
    return null;
  }

  const position = simulation.helicopterPosition;

  return (
    <Marker
      markerImg={helicopterIcon}
      position={position}
      width={0.08}
      height={0.08}
      anchorX={0.5}
      anchorY={0.5}
      dragPlane={dragPlane}
    />
  );
});

