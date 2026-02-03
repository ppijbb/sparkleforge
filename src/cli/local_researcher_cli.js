/**
 * Local Researcher CLI - Main Interface
 * 
 * This module provides the main interface for the Local Researcher CLI
 * that integrates with the Python backend.
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs-extra');

class LocalResearcherCLI {
    constructor() {
        this.pythonScript = path.join(__dirname, 'gemini_integration.py');
        this.isInitialized = false;
    }
    
    /**
     * Start a new research project
     * @param {Object} researchRequest - Research request object
     * @returns {Promise<Object>} Research result
     */
    async startResearch(researchRequest) {
        try {
            // Validate request
            if (!researchRequest.topic || researchRequest.topic.trim().length === 0) {
                throw new Error('Research topic is required');
            }
            
            // Call Python backend
            const result = await this._callPythonBackend('start_research', researchRequest);
            
            return {
                success: true,
                research_id: result.research_id || this._generateResearchId(researchRequest.topic),
                message: 'Research started successfully'
            };
            
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }
    
    /**
     * Get research status
     * @param {string} researchId - Research ID
     * @returns {Promise<Object>} Research status
     */
    async getResearchStatus(researchId) {
        try {
            const result = await this._callPythonBackend('get_status', { research_id: researchId });
            return result;
        } catch (error) {
            return null;
        }
    }
    
    /**
     * Get active research projects
     * @returns {Promise<Array>} List of active research
     */
    async getActiveResearch() {
        try {
            const result = await this._callPythonBackend('get_active_research', {});
            return result.active_research || [];
        } catch (error) {
            return [];
        }
    }
    
    /**
     * Get research list with optional filtering
     * @param {Object} options - Filter options
     * @returns {Promise<Array>} List of research projects
     */
    async getResearchList(options = {}) {
        try {
            const result = await this._callPythonBackend('get_research_list', options);
            return result.research_list || [];
        } catch (error) {
            return [];
        }
    }
    
    /**
     * Cancel a research project
     * @param {string} researchId - Research ID to cancel
     * @returns {Promise<boolean>} Success status
     */
    async cancelResearch(researchId) {
        try {
            const result = await this._callPythonBackend('cancel_research', { research_id: researchId });
            return result.success || false;
        } catch (error) {
            return false;
        }
    }
    
    /**
     * Initialize the research system
     * @returns {Promise<boolean>} Initialization success
     */
    async initialize() {
        try {
            // Check if Python script exists
            if (!fs.existsSync(this.pythonScript)) {
                throw new Error('Python backend script not found');
            }
            
            // Test Python backend
            const result = await this._callPythonBackend('ping', {});
            
            if (result.status === 'ok') {
                this.isInitialized = true;
                return true;
            } else {
                throw new Error('Python backend not responding');
            }
            
        } catch (error) {
            console.error('Failed to initialize research system:', error.message);
            return false;
        }
    }
    
    /**
     * Call Python backend with specified method and parameters
     * @param {string} method - Method to call
     * @param {Object} params - Parameters to pass
     * @returns {Promise<Object>} Python backend response
     */
    async _callPythonBackend(method, params) {
        return new Promise((resolve, reject) => {
            const pythonProcess = spawn('python3', [this.pythonScript, method, JSON.stringify(params)]);
            
            let stdout = '';
            let stderr = '';
            
            pythonProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            
            pythonProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            
            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(stdout);
                        resolve(result);
                    } catch (error) {
                        reject(new Error('Invalid response from Python backend'));
                    }
                } else {
                    reject(new Error(`Python backend failed with code ${code}: ${stderr}`));
                }
            });
            
            pythonProcess.on('error', (error) => {
                reject(new Error(`Failed to start Python backend: ${error.message}`));
            });
            
            // Set timeout
            setTimeout(() => {
                pythonProcess.kill();
                reject(new Error('Python backend timeout'));
            }, 30000); // 30 seconds timeout
        });
    }
    
    /**
     * Generate a unique research ID
     * @param {string} topic - Research topic
     * @returns {string} Research ID
     */
    _generateResearchId(topic) {
        const timestamp = Date.now();
        const topicSlug = topic.toLowerCase().replace(/[^a-z0-9]/g, '_').substring(0, 20);
        return `research_${topicSlug}_${timestamp}`;
    }
    
    /**
     * Check if the system is initialized
     * @returns {boolean} Initialization status
     */
    isReady() {
        return this.isInitialized;
    }
    
    /**
     * Get system information
     * @returns {Object} System information
     */
    getSystemInfo() {
        return {
            initialized: this.isInitialized,
            python_script: this.pythonScript,
            node_version: process.version,
            platform: process.platform,
            arch: process.arch
        };
    }
}

module.exports = { LocalResearcherCLI };
