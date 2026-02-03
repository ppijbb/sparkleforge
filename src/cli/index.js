#!/usr/bin/env node

/**
 * Local Researcher CLI - Main Entry Point
 * 
 * This is the main entry point for the Local Researcher CLI that integrates
 * with Gemini CLI for seamless research operations.
 */

const { program } = require('commander');
const inquirer = require('inquirer');
const chalk = require('chalk');
const ora = require('ora');
const path = require('path');
const fs = require('fs-extra');
const { spawn } = require('child_process');

// Import local modules
const { LocalResearcherCLI } = require('./local_researcher_cli');
const { ConfigManager } = require('../utils/config_manager');
const { Logger } = require('../utils/logger');

class MainCLI {
    constructor() {
        this.program = program;
        this.configManager = new ConfigManager();
        this.logger = new Logger('main_cli');
        this.localResearcher = new LocalResearcherCLI();
        
        this.setupCommands();
    }
    
    setupCommands() {
        // Main research command
        this.program
            .command('research <topic>')
            .description('Start a new research project')
            .option('-d, --domain <domain>', 'Specify research domain (technology, science, business, general)')
            .option('--depth <depth>', 'Research depth (basic, standard, comprehensive)', 'standard')
            .option('--sources <sources>', 'Comma-separated list of sources')
            .option('--format <format>', 'Output format (markdown, pdf, html)', 'markdown')
            .option('--interactive', 'Run in interactive mode')
            .action(async (topic, options) => {
                await this.handleResearchCommand(topic, options);
            });
        
        // Status command
        this.program
            .command('status [research_id]')
            .description('Check research status')
            .action(async (researchId) => {
                await this.handleStatusCommand(researchId);
            });
        
        // List command
        this.program
            .command('list')
            .description('List all research projects')
            .option('--active', 'Show only active research')
            .option('--completed', 'Show only completed research')
            .option('--failed', 'Show only failed research')
            .action(async (options) => {
                await this.handleListCommand(options);
            });
        
        // Cancel command
        this.program
            .command('cancel <research_id>')
            .description('Cancel a research project')
            .action(async (researchId) => {
                await this.handleCancelCommand(researchId);
            });
        
        // Setup command
        this.program
            .command('setup')
            .description('Setup Local Researcher environment')
            .action(async () => {
                await this.handleSetupCommand();
            });
        
        // Config command
        this.program
            .command('config')
            .description('Manage configuration')
            .option('--show', 'Show current configuration')
            .option('--edit', 'Edit configuration')
            .option('--reset', 'Reset to default configuration')
            .action(async (options) => {
                await this.handleConfigCommand(options);
            });
        
        // Help command
        this.program
            .command('help')
            .description('Show help information')
            .action(() => {
                this.showHelp();
            });
        
        // Version command
        this.program
            .version(require('../../package.json').version, '-v, --version');
        
        // Default action
        this.program
            .action(async () => {
                await this.runInteractiveMode();
            });
    }
    
    async handleResearchCommand(topic, options) {
        const spinner = ora('Starting research...').start();
        
        try {
            // Validate topic
            if (!topic || topic.trim().length === 0) {
                spinner.fail('Research topic is required');
                return;
            }
            
            // Prepare research request
            const researchRequest = {
                topic: topic.trim(),
                domain: options.domain || 'general',
                depth: options.depth,
                sources: options.sources ? options.sources.split(',') : [],
                format: options.format,
                interactive: options.interactive
            };
            
            spinner.text = 'Initializing research system...';
            
            // Start research
            const result = await this.localResearcher.startResearch(researchRequest);
            
            if (result.success) {
                spinner.succeed(`Research started: ${result.research_id}`);
                console.log(chalk.green(`Topic: ${topic}`));
                console.log(chalk.blue(`Domain: ${researchRequest.domain}`));
                console.log(chalk.blue(`Depth: ${researchRequest.depth}`));
                
                if (options.interactive) {
                    console.log(chalk.yellow('\nRunning in interactive mode...'));
                    await this.runInteractiveMode();
                }
            } else {
                spinner.fail(`Research failed: ${result.error}`);
            }
            
        } catch (error) {
            spinner.fail(`Error: ${error.message}`);
            this.logger.error('Research command failed', error);
        }
    }
    
    async handleStatusCommand(researchId) {
        try {
            if (researchId) {
                // Get specific research status
                const status = await this.localResearcher.getResearchStatus(researchId);
                if (status) {
                    this.displayResearchStatus(status);
                } else {
                    console.log(chalk.red(`Research not found: ${researchId}`));
                }
            } else {
                // Get all active research status
                const activeResearch = await this.localResearcher.getActiveResearch();
                if (activeResearch.length === 0) {
                    console.log(chalk.yellow('No active research projects'));
                } else {
                    this.displayResearchList(activeResearch);
                }
            }
        } catch (error) {
            console.log(chalk.red(`Error: ${error.message}`));
            this.logger.error('Status command failed', error);
        }
    }
    
    async handleListCommand(options) {
        try {
            const researchList = await this.localResearcher.getResearchList(options);
            this.displayResearchList(researchList);
        } catch (error) {
            console.log(chalk.red(`Error: ${error.message}`));
            this.logger.error('List command failed', error);
        }
    }
    
    async handleCancelCommand(researchId) {
        const spinner = ora('Cancelling research...').start();
        
        try {
            const success = await this.localResearcher.cancelResearch(researchId);
            
            if (success) {
                spinner.succeed(`Research cancelled: ${researchId}`);
            } else {
                spinner.fail(`Failed to cancel research: ${researchId}`);
            }
        } catch (error) {
            spinner.fail(`Error: ${error.message}`);
            this.logger.error('Cancel command failed', error);
        }
    }
    
    async handleSetupCommand() {
        const spinner = ora('Setting up Local Researcher...').start();
        
        try {
            // Check prerequisites
            spinner.text = 'Checking prerequisites...';
            await this.checkPrerequisites();
            
            // Install dependencies
            spinner.text = 'Installing dependencies...';
            await this.installDependencies();
            
            // Setup configuration
            spinner.text = 'Setting up configuration...';
            await this.setupConfiguration();
            
            // Test integration
            spinner.text = 'Testing integration...';
            await this.testIntegration();
            
            spinner.succeed('Local Researcher setup completed successfully!');
            console.log(chalk.green('\nYou can now start using Local Researcher:'));
            console.log(chalk.blue('  gemini research "your research topic"'));
            
        } catch (error) {
            spinner.fail(`Setup failed: ${error.message}`);
            this.logger.error('Setup command failed', error);
        }
    }
    
    async handleConfigCommand(options) {
        try {
            if (options.show) {
                const config = this.configManager.getConfig();
                console.log(chalk.blue('Current Configuration:'));
                console.log(JSON.stringify(config, null, 2));
            } else if (options.edit) {
                await this.editConfiguration();
            } else if (options.reset) {
                await this.resetConfiguration();
            } else {
                console.log(chalk.yellow('Use --show, --edit, or --reset option'));
            }
        } catch (error) {
            console.log(chalk.red(`Error: ${error.message}`));
            this.logger.error('Config command failed', error);
        }
    }
    
    async checkPrerequisites() {
        // Check Node.js version
        const nodeVersion = process.version;
        const requiredVersion = '20.0.0';
        
        if (this.compareVersions(nodeVersion, requiredVersion) < 0) {
            throw new Error(`Node.js ${requiredVersion} or higher is required. Current: ${nodeVersion}`);
        }
        
        // Check Python version
        const pythonVersion = await this.getPythonVersion();
        const requiredPythonVersion = '3.11.0';
        
        if (this.compareVersions(pythonVersion, requiredPythonVersion) < 0) {
            throw new Error(`Python ${requiredPythonVersion} or higher is required. Current: ${pythonVersion}`);
        }
        
        // Check Gemini CLI
        const geminiInstalled = await this.checkGeminiCLI();
        if (!geminiInstalled) {
            throw new Error('Gemini CLI is not installed. Please install it first.');
        }
    }
    
    async installDependencies() {
        // Install Node.js dependencies
        await this.runCommand('npm', ['install']);
        
        // Install Python dependencies
        await this.runCommand('pip', ['install', '-r', 'requirements.txt']);
    }
    
    async setupConfiguration() {
        const configPath = path.join(process.cwd(), 'configs', 'config.yaml');
        const envPath = path.join(process.cwd(), '.env');
        
        // Create .env file if it doesn't exist
        if (!fs.existsSync(envPath)) {
            const envTemplate = this.getEnvTemplate();
            await fs.writeFile(envPath, envTemplate);
            console.log(chalk.yellow('Created .env file. Please configure your API keys.'));
        }
        
        // Create output directories
        const outputDir = path.join(process.cwd(), 'outputs');
        const logsDir = path.join(process.cwd(), 'logs');
        const dataDir = path.join(process.cwd(), 'data');
        
        await fs.ensureDir(outputDir);
        await fs.ensureDir(logsDir);
        await fs.ensureDir(dataDir);
    }
    
    async testIntegration() {
        // Test Gemini CLI integration
        const geminiTest = await this.testGeminiCLI();
        if (!geminiTest) {
            throw new Error('Gemini CLI integration test failed');
        }
        
        // Test Open Deep Research integration
        const odrTest = await this.testOpenDeepResearch();
        if (!odrTest) {
            throw new Error('Open Deep Research integration test failed');
        }
    }
    
    async runInteractiveMode() {
        console.log(chalk.blue.bold('\nLocal Researcher - Interactive Mode'));
        console.log(chalk.gray('Type "help" for available commands or "exit" to quit.\n'));
        
        while (true) {
            try {
                const { command } = await inquirer.prompt([
                    {
                        type: 'input',
                        name: 'command',
                        message: chalk.green('local-researcher> '),
                        prefix: ''
                    }
                ]);
                
                if (command.toLowerCase() === 'exit' || command.toLowerCase() === 'quit') {
                    console.log(chalk.blue('Goodbye!'));
                    break;
                }
                
                if (command.toLowerCase() === 'help') {
                    this.showHelp();
                    continue;
                }
                
                // Parse and execute command
                const args = command.split(' ');
                const cmd = args[0];
                const cmdArgs = args.slice(1);
                
                // Execute command using program
                await this.program.parseAsync(['node', 'index.js', cmd, ...cmdArgs]);
                
            } catch (error) {
                console.log(chalk.red(`Error: ${error.message}`));
            }
        }
    }
    
    displayResearchStatus(status) {
        console.log(chalk.blue(`\nResearch Status: ${status.request_id}`));
        console.log(chalk.green(`Topic: ${status.topic}`));
        console.log(chalk.yellow(`Status: ${status.status}`));
        console.log(chalk.cyan(`Progress: ${status.progress.toFixed(1)}%`));
        
        if (status.report_path) {
            console.log(chalk.green(`Report: ${status.report_path}`));
        }
        
        if (status.error_message) {
            console.log(chalk.red(`Error: ${status.error_message}`));
        }
    }
    
    displayResearchList(researchList) {
        if (researchList.length === 0) {
            console.log(chalk.yellow('No research projects found'));
            return;
        }
        
        console.log(chalk.blue('\nResearch Projects:'));
        console.log('─'.repeat(80));
        
        researchList.forEach(research => {
            const statusColor = research.status === 'completed' ? 'green' : 
                              research.status === 'failed' ? 'red' : 'yellow';
            
            console.log(chalk.cyan(`ID: ${research.request_id}`));
            console.log(chalk.green(`Topic: ${research.topic}`));
            console.log(chalk[statusColor](`Status: ${research.status}`));
            console.log(chalk.cyan(`Progress: ${research.progress.toFixed(1)}%`));
            
            if (research.report_path) {
                console.log(chalk.green(`Report: ${research.report_path}`));
            }
            
            console.log('─'.repeat(80));
        });
    }
    
    showHelp() {
        console.log(chalk.blue.bold('\nLocal Researcher - Available Commands:'));
        console.log('');
        console.log(chalk.green('  research <topic>     Start a new research project'));
        console.log(chalk.green('  status [id]         Check research status'));
        console.log(chalk.green('  list                List all research projects'));
        console.log(chalk.green('  cancel <id>         Cancel a research project'));
        console.log(chalk.green('  setup               Setup Local Researcher'));
        console.log(chalk.green('  config              Manage configuration'));
        console.log(chalk.green('  help                Show this help'));
        console.log('');
        console.log(chalk.gray('For more information, visit: https://github.com/your-org/local-researcher'));
    }
    
    // Utility methods
    async getPythonVersion() {
        try {
            const result = await this.runCommand('python', ['--version']);
            return result.stdout.trim().replace('Python ', '');
        } catch (error) {
            throw new Error('Python is not installed or not in PATH');
        }
    }
    
    async checkGeminiCLI() {
        try {
            await this.runCommand('gemini', ['--version']);
            return true;
        } catch (error) {
            return false;
        }
    }
    
    async testGeminiCLI() {
        try {
            await this.runCommand('gemini', ['--help']);
            return true;
        } catch (error) {
            return false;
        }
    }
    
    async testOpenDeepResearch() {
        // Test implementation required
        throw new Error("testOpenDeepResearch not implemented");
    }
    
    async runCommand(command, args) {
        return new Promise((resolve, reject) => {
            const child = spawn(command, args, { stdio: 'pipe' });
            let stdout = '';
            let stderr = '';
            
            child.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            
            child.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            
            child.on('close', (code) => {
                if (code === 0) {
                    resolve({ stdout, stderr });
                } else {
                    reject(new Error(`Command failed: ${stderr}`));
                }
            });
        });
    }
    
    compareVersions(version1, version2) {
        const v1 = version1.replace(/^v/, '').split('.').map(Number);
        const v2 = version2.split('.').map(Number);
        
        for (let i = 0; i < Math.max(v1.length, v2.length); i++) {
            const num1 = v1[i] || 0;
            const num2 = v2[i] || 0;
            
            if (num1 > num2) return 1;
            if (num1 < num2) return -1;
        }
        
        return 0;
    }
    
    getEnvTemplate() {
        return `# Local Researcher Environment Variables
# Copy this file to .env and configure your API keys

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORG_ID=your_openai_org_id_here

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Configuration
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_PROJECT_ID=your_google_project_id_here

# News API Configuration
NEWSAPI_KEY=your_newsapi_key_here

# Tavily Configuration
TAVILY_API_KEY=your_tavily_api_key_here

# Perplexity Configuration
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Application Configuration
NODE_ENV=production
LOG_LEVEL=INFO
`;
    }
}

// Initialize and run CLI
const cli = new MainCLI();
cli.program.parse(process.argv);

module.exports = { MainCLI }; 