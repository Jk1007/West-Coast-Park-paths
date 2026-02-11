import React from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Dimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

export default function SimOverlay({ stats, onReset, onUpdateWind, wind }) {
    return (
        <View style={styles.container}>
            {/* Top Bar: Status */}
            <View style={[styles.statusBar, stats.activeIncidents > 0 ? styles.statusDanger : styles.statusClear]}>
                <Text style={styles.statusText}>Active Incidents: {stats.activeIncidents}</Text>
                <Text style={styles.statusText}>Safety Index: {stats.safetyIndex}%</Text>
            </View>

            {/* Bottom Controls */}
            <View style={styles.controls}>
                <View style={styles.windControl}>
                    <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
                        <View>
                            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                                <Text style={styles.controlLabel}>Live Wind: {wind.speed.toFixed(1)} km/h</Text>
                                {wind.direction !== undefined && (
                                    <View style={{
                                        marginLeft: 8,
                                        transform: [{ rotate: `${wind.direction + 180}deg` }] // Point with the wind flow
                                    }}>
                                        <Ionicons name="arrow-up" size={20} color="#fff" />
                                    </View>
                                )}
                            </View>
                            <Text style={[styles.controlLabel, { fontSize: 10, color: '#aaa', textAlign: 'left' }]}>Weighted: Clementi/Sentosa/Banyan</Text>
                        </View>
                        <TouchableOpacity style={[styles.buttonSmall, { backgroundColor: '#2980b9' }]} onPress={() => onUpdateWind()}>
                            <Text style={styles.buttonText}>Refresh</Text>
                        </TouchableOpacity>
                    </View>
                </View>

                <TouchableOpacity style={styles.resetButton} onPress={onReset}>
                    <Text style={styles.resetText}>Reset Simulation</Text>
                </TouchableOpacity>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        ...StyleSheet.absoluteFillObject,
        justifyContent: 'space-between',
        padding: 20,
        marginTop: 40,
        pointerEvents: 'box-none', // Allow touches to pass through to map
    },
    statusBar: {
        padding: 15,
        borderRadius: 8,
        alignItems: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.3,
        shadowRadius: 3,
        elevation: 5,
    },
    statusClear: {
        backgroundColor: 'rgba(46, 204, 113, 0.9)',
    },
    statusDanger: {
        backgroundColor: 'rgba(231, 76, 60, 0.9)',
    },
    statusText: {
        color: '#fff',
        fontSize: 18,
        fontWeight: 'bold',
    },
    controls: {
        backgroundColor: 'rgba(0,0,0,0.7)',
        padding: 15,
        borderRadius: 12,
        pointerEvents: 'auto',
    },
    windControl: {
        marginBottom: 10,
    },
    controlLabel: {
        color: '#fff',
        marginBottom: 5,
        textAlign: 'center',
    },
    buttonRow: {
        flexDirection: 'row',
        justifyContent: 'space-around',
    },
    buttonSmall: {
        backgroundColor: '#3498db',
        padding: 8,
        borderRadius: 5,
        marginHorizontal: 5,
    },
    buttonText: {
        color: '#fff',
        fontSize: 12,
        fontWeight: 'bold',
    },
    resetButton: {
        backgroundColor: '#f39c12',
        padding: 12,
        borderRadius: 8,
        alignItems: 'center',
    },
    resetText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: 'bold',
    },
});
