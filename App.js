import React, { useEffect, useState, useRef } from 'react';
import { StyleSheet, View } from 'react-native';
import MapView, { Marker, Circle, PROVIDER_GOOGLE, PROVIDER_DEFAULT } from 'react-native-maps';
import { SimulationController } from './src/simulation/SimulationController';
import { PARK_CENTER } from './src/data/ParkData';
import SimOverlay from './src/components/SimOverlay';

// Initial Region for React Native Maps
const INITIAL_REGION = {
  latitude: PARK_CENTER[1],
  longitude: PARK_CENTER[0],
  latitudeDelta: 0.005,
  longitudeDelta: 0.005,
};

export default function App() {
  const [simState, setSimState] = useState({
    agents: [],
    stats: { activeIncidents: 0, safetyIndex: 100 },
    wind: { speed: 5, direction: 45 },
    incidents: [],
  });

  // Lazy initialization to prevent constructor running on every render
  const simController = useRef(null);
  if (!simController.current) {
    simController.current = new SimulationController();
  }

  const animationFrameId = useRef(null);

  useEffect(() => {
    let lastTime = Date.now();

    const loop = () => {
      const now = Date.now();
      const dt = (now - lastTime) / 1000;
      lastTime = now;

      simController.current.update(dt);

      setSimState({
        agents: [...simController.current.agents],
        stats: simController.current.getStats(),
        wind: simController.current.wind,
        incidents: [...simController.current.incidents],
      });

      animationFrameId.current = requestAnimationFrame(loop);
      // Note: For 100ms interval strictness as per prompt, we could use setInterval, 
      // but requestAnimationFrame is smoother for UI. 
      // The physics update uses 'dt' so it stays correct regardless of frame rate.
    };

    loop();

    return () => {
      if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
    };
  }, []);

  const handleLongPress = (e) => {
    // react-native-maps event: e.nativeEvent.coordinate
    const coords = e.nativeEvent.coordinate;
    // Controller expects [lon, lat] array
    simController.current.addIncident([coords.longitude, coords.latitude]);
  };

  const handleReset = () => {
    simController.current.reset();
  };

  const handleWindUpdate = () => {
    simController.current.updateWindFromAPI(true); // Force update
  };

  return (
    <View style={styles.container}>
      <MapView
        style={styles.map}
        initialRegion={INITIAL_REGION}
        showsBuildings={true}
        showsIndoors={false}
        pitchEnabled={true}
        // Constrain Region
        minZoomLevel={15} // Prevent zooming out too far
        maxZoomLevel={20}
        // region={INITIAL_REGION} // Optional: STRICT lock, but users might want to pan slightly. 
        // prompt says "wiggle room of about 2km". 
        // We can do this by just setting minZoomLevel high enough so they can't see whole world.
        onLongPress={handleLongPress}
      >
        {/* Render Agents as Markers */}
        {simState.agents.map(agent => (
          <Marker
            key={agent.id}
            coordinate={{
              latitude: agent.position[1],
              longitude: agent.position[0],
            }}
            anchor={{ x: 0.5, y: 0.5 }}
            flat={true} // Billboard effect
          >
            {/* Custom Marker View for Performance (Simple View) */}
            <View style={[
              styles.agentDot,
              { backgroundColor: agent.state === 'IDLE' ? '#3498db' : '#e74c3c' }
            ]} />
          </Marker>
        ))}

        {/* Render Incidents as Circles */}
        {simState.incidents.map(inc => (
          <Circle
            key={inc.id}
            center={{
              latitude: inc.position[1],
              longitude: inc.position[0],
            }}
            radius={inc.radius}
            fillColor="rgba(46, 204, 113, 0.5)"
            strokeColor="rgba(46, 204, 113, 0.8)"
            zIndex={1}
          />
        ))}

      </MapView>

      <SimOverlay
        stats={simState.stats}
        wind={simState.wind}
        onReset={handleReset}
        onUpdateWind={handleWindUpdate}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  map: {
    flex: 1,
  },
  agentDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    borderWidth: 1,
    borderColor: '#fff',
  },
});
