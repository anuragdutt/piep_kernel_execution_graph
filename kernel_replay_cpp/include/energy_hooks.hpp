#pragma once

namespace energy {

/**
 * Abstract interface for energy measurement probes.
 * 
 * This is a placeholder that will be replaced with the actual
 * physical probe interface later.
 */
class EnergyProbe {
public:
    virtual ~EnergyProbe() = default;
    
    /**
     * Start energy measurement
     */
    virtual void start_measurement() = 0;
    
    /**
     * Read accumulated energy since start_measurement() was called
     * 
     * @return Energy consumed in Joules
     */
    virtual double read_energy_joules() = 0;
    
    /**
     * Reset the energy counter
     */
    virtual void reset() = 0;
};

/**
 * Dummy implementation for testing without physical probe
 */
class DummyProbe : public EnergyProbe {
public:
    void start_measurement() override {
        // No-op
    }
    
    double read_energy_joules() override {
        // Return 0 - replace with actual probe reading
        return 0.0;
    }
    
    void reset() override {
        // No-op
    }
};

/**
 * Factory function to create energy probe
 * 
 * For now returns DummyProbe. Will be updated to create actual probe
 * when physical interface is available.
 */
inline EnergyProbe* create_probe() {
    return new DummyProbe();
}

} // namespace energy
