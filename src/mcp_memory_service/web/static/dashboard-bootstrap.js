import ForceGraph3D from 'https://esm.sh/3d-force-graph@1.80.0?external=three';
import * as THREE from 'three';
import { UnrealBloomPass } from 'https://esm.sh/three@0.184.0/examples/jsm/postprocessing/UnrealBloomPass.js?external=three';

window.ForceGraph3D = ForceGraph3D;
window.THREE = THREE;
window.__UnrealBloomPass = UnrealBloomPass;
window.dispatchEvent(new CustomEvent('forcegraph3d-ready'));
