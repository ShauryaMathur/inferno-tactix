import React, { useCallback, useEffect, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls , PerspectiveCamera } from "@react-three/drei";
import { Provider } from "mobx-react";
import { useStores } from "../../use-stores";
import { DEFAULT_UP, PLANE_WIDTH, planeHeight } from "./helpers";
import { Terrain } from "./terrain";
import { SparksContainer } from "./spark";
import * as THREE from "three";
import { FireLineMarkersContainer } from "./fire-line-marker";
import { TownMarkersContainer } from "./town-marker";
import Shutterbug from "shutterbug";

// This needs to be a separate component, as useThree depends on context provided by <Canvas> component.
const ShutterbugSupport = () => {
  const { gl, scene, camera } = useThree();
  useEffect(() => {
    const render = () => {
      gl.render(scene, camera);
    };
    Shutterbug.on("saycheese", render);
    return () => Shutterbug.off("saycheese", render);
  }, [gl, scene, camera]);
  return null;
};

export const View3d = () => {
  const stores = useStores();
  const simulation = stores.simulation;
  const ui = stores.ui;
  const cameraPos: [number, number, number] = [PLANE_WIDTH * 0.5, planeHeight(simulation) * -1.5, PLANE_WIDTH * 1.5];
  const terrainRef = useRef<THREE.Mesh>(null);

  return (
    /* eslint-disable react/no-unknown-property */
    // See: https://github.com/jsx-eslint/eslint-plugin-react/issues/3423
    // flat=true disables tone mapping that is not a default in threejs, but is enabled by default in react-three-fiber.
    // It makes textures match colors in the original image.
    <Canvas
  // camera={{ manual: true }}
  // flat={true}
  // // ① expose alpha & antialias on the WebGLRenderer
  // gl={{ antialias: true, alpha: true }}
  // // ② run code immediately after the renderer+scene are created
  // onCreated={({ gl, scene }) => {
  //   // make the clear color fully transparent
  //   gl.setClearColor(0x000000, 0);

  //   // remove the default gray background
  //   scene.background = null;

  //   // load a simple cube‐map sky (6 images in /public/textures/)
  //   const loader = new THREE.CubeTextureLoader();
  //   const envMap = loader.load([
  //     '/textures/sky_px.jpg',
  //     '/textures/sky_nx.jpg',
  //     '/textures/sky_py.jpg',
  //     '/textures/sky_ny.jpg',
  //     '/textures/sky_pz.jpg',
  //     '/textures/sky_nz.jpg',
  //   ]);

  //   // assign it as both background and environment
  //   scene.environment = envMap;
  //   scene.background  = envMap;
  // }}
>
      {/* Why do we need to setup provider again? No idea. It seems that components inside Canvas don't have
          access to MobX stores anymore. */}
      <Provider stores={stores}>
        <PerspectiveCamera makeDefault={true} fov={33} position={cameraPos} up={DEFAULT_UP}/>
        <OrbitControls
          target={[PLANE_WIDTH * 0.5, planeHeight(simulation) * 0.5, 0.2]}
          enableDamping={true}
          enableRotate={!ui.dragging} // disable rotation when something is being dragged
          enablePan={false}
          rotateSpeed={0.5}
          zoomSpeed={0.5}
          minDistance={0.8}
          maxDistance={5}
          maxPolarAngle={Math.PI * 0.4}
          minAzimuthAngle={-Math.PI * 0.25}
          maxAzimuthAngle={Math.PI * 0.25}
        />
        <hemisphereLight args={[0xC6C2B6, 0x3A403B, 1.2]} up={DEFAULT_UP}/>
        {/* <directionalLight
  args={[0xffffff, 0.8]}
  position={[-100, 100, -100]}
/> */}
        <Terrain ref={terrainRef}/>
        <SparksContainer dragPlane={terrainRef}/>
        <FireLineMarkersContainer dragPlane={terrainRef}/>
        {/* <TownMarkersContainer/> */}
        <ShutterbugSupport/>
      </Provider>
    </Canvas>
    /* eslint-enable react/no-unknown-property */
  );
};
