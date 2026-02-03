/**
 * A2UI JavaScript Renderer for Streamlit
 * 
 * Renders A2UI (Agent-to-User Interface) JSON into interactive HTML components.
 * Based on A2UI 0.9 specification.
 * 
 * @license Apache-2.0
 */

class A2UIRenderer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error(`Container with id "${containerId}" not found`);
      return;
    }
    
    this.surfaces = new Map(); // surfaceId -> Surface
    this.dataModels = new Map(); // surfaceId -> DataModel
    this.components = new Map(); // componentId -> DOM Element
    this.themes = new Map(); // surfaceId -> Theme
    
    // Initialize container
    this.container.style.width = '100%';
    this.container.style.minHeight = '200px';
  }
  
  /**
   * Render an A2UI message
   * @param {Object} message - A2UI message (createSurface, updateComponents, updateDataModel, deleteSurface)
   */
  render(message) {
    try {
      if (message.createSurface) {
        this.createSurface(message.createSurface);
      } else if (message.updateComponents) {
        this.updateComponents(message.updateComponents);
      } else if (message.updateDataModel) {
        this.updateDataModel(message.updateDataModel);
      } else if (message.deleteSurface) {
        this.deleteSurface(message.deleteSurface);
      } else {
        console.warn('Unknown A2UI message type:', message);
      }
    } catch (error) {
      console.error('Error rendering A2UI message:', error);
    }
  }
  
  /**
   * Create a new surface
   * @param {Object} config - Surface configuration
   */
  createSurface(config) {
    const { surfaceId, catalogId } = config;
    
    if (this.surfaces.has(surfaceId)) {
      console.warn(`Surface ${surfaceId} already exists, recreating...`);
      this.deleteSurface({ surfaceId });
    }
    
    // Create surface container
    const surfaceElement = document.createElement('div');
    surfaceElement.id = `a2ui-surface-${surfaceId}`;
    surfaceElement.className = 'a2ui-surface';
    surfaceElement.style.width = '100%';
    this.container.appendChild(surfaceElement);
    
    // Initialize data model
    this.dataModels.set(surfaceId, {});
    
    // Store surface
    this.surfaces.set(surfaceId, {
      element: surfaceElement,
      catalogId: catalogId,
      rootComponentId: null
    });
    
    console.log(`Created surface: ${surfaceId}`);
  }
  
  /**
   * Update components in a surface
   * @param {Object} update - Component update message
   */
  updateComponents(update) {
    const { surfaceId, components } = update;
    
    const surface = this.surfaces.get(surfaceId);
    if (!surface) {
      console.error(`Surface ${surfaceId} not found`);
      return;
    }
    
    // Clear existing components
    surface.element.innerHTML = '';
    this.components.clear();
    
    // Find root component
    const rootComponent = components.find(c => c.id === 'root');
    if (!rootComponent) {
      console.error('Root component not found');
      return;
    }
    
    surface.rootComponentId = rootComponent.id;
    
    // Build component map
    const componentMap = new Map();
    components.forEach(comp => {
      componentMap.set(comp.id, comp);
    });
    
    // Render root component
    const rootElement = this.renderComponent(rootComponent, componentMap, surfaceId);
    if (rootElement) {
      surface.element.appendChild(rootElement);
    }
  }
  
  /**
   * Update data model for a surface
   * @param {Object} update - Data model update message
   */
  updateDataModel(update) {
    const { surfaceId, path, op, value } = update;
    
    const dataModel = this.dataModels.get(surfaceId);
    if (!dataModel) {
      console.error(`Data model for surface ${surfaceId} not found`);
      return;
    }
    
    const operation = op || 'replace';
    
    if (operation === 'replace' || operation === 'add') {
      if (path === '/' || !path) {
        // Update entire data model
        Object.assign(dataModel, value);
      } else {
        // Update specific path
        this.setNestedValue(dataModel, path, value);
      }
    } else if (operation === 'remove') {
      if (path && path !== '/') {
        this.deleteNestedValue(dataModel, path);
      }
    }
    
    // Re-render components that depend on this data
    this.updateDataBindings(surfaceId);
  }
  
  /**
   * Delete a surface
   * @param {Object} config - Delete configuration
   */
  deleteSurface(config) {
    const { surfaceId } = config;
    
    const surface = this.surfaces.get(surfaceId);
    if (surface && surface.element) {
      surface.element.remove();
    }
    
    this.surfaces.delete(surfaceId);
    this.dataModels.delete(surfaceId);
    this.themes.delete(surfaceId);
    
    // Remove all components for this surface
    for (const [id, element] of this.components.entries()) {
      if (id.startsWith(`${surfaceId}-`)) {
        this.components.delete(id);
      }
    }
  }
  
  /**
   * Render a component
   * @param {Object} component - Component definition
   * @param {Map} componentMap - Map of all components
   * @param {string} surfaceId - Surface ID
   * @returns {HTMLElement} - Rendered DOM element
   */
  renderComponent(component, componentMap, surfaceId) {
    if (!component || !component.component) {
      return null;
    }
    
    const componentType = component.component;
    const elementId = `${surfaceId}-${component.id}`;
    
    let element = null;
    
    try {
      switch (componentType) {
        case 'Text':
          element = this.renderText(component, surfaceId);
          break;
        case 'Image':
          element = this.renderImage(component, surfaceId);
          break;
        case 'Icon':
          element = this.renderIcon(component, surfaceId);
          break;
        case 'Button':
          element = this.renderButton(component, componentMap, surfaceId);
          break;
        case 'Card':
          element = this.renderCard(component, componentMap, surfaceId);
          break;
        case 'Row':
          element = this.renderRow(component, componentMap, surfaceId);
          break;
        case 'Column':
          element = this.renderColumn(component, componentMap, surfaceId);
          break;
        case 'List':
          element = this.renderList(component, componentMap, surfaceId);
          break;
        case 'TextField':
          element = this.renderTextField(component, surfaceId);
          break;
        case 'CheckBox':
          element = this.renderCheckBox(component, surfaceId);
          break;
        case 'Slider':
          element = this.renderSlider(component, surfaceId);
          break;
        case 'ChoicePicker':
          element = this.renderChoicePicker(component, surfaceId);
          break;
        case 'Tabs':
          element = this.renderTabs(component, componentMap, surfaceId);
          break;
        case 'Modal':
          element = this.renderModal(component, componentMap, surfaceId);
          break;
        case 'Divider':
          element = this.renderDivider(component);
          break;
        case 'Video':
          element = this.renderVideo(component, surfaceId);
          break;
        case 'AudioPlayer':
          element = this.renderAudioPlayer(component, surfaceId);
          break;
        default:
          console.warn(`Unknown component type: ${componentType}`);
          element = document.createElement('div');
          element.textContent = `[Unknown component: ${componentType}]`;
      }
      
      if (element) {
        element.id = elementId;
        element.className = `a2ui-component a2ui-${componentType.toLowerCase()}`;
        
        // Apply weight (flex-grow)
        if (component.weight) {
          element.style.flexGrow = component.weight;
        }
        
        this.components.set(elementId, element);
      }
    } catch (error) {
      console.error(`Error rendering component ${component.id}:`, error);
      element = document.createElement('div');
      element.textContent = `[Error rendering ${componentType}]`;
    }
    
    return element;
  }
  
  /**
   * Resolve a value (string or path)
   * @param {string|Object} value - Value or path object
   * @param {string} surfaceId - Surface ID
   * @returns {*} - Resolved value
   */
  resolveValue(value, surfaceId) {
    if (typeof value === 'string') {
      return value;
    } else if (value && typeof value === 'object' && value.path) {
      const dataModel = this.dataModels.get(surfaceId);
      if (dataModel) {
        return this.getNestedValue(dataModel, value.path);
      }
    }
    return value;
  }
  
  /**
   * Get nested value from object by path
   * @param {Object} obj - Object
   * @param {string} path - Path (e.g., '/user/name' or 'chart.items[0].label')
   * @returns {*} - Value
   */
  getNestedValue(obj, path) {
    if (!path || path === '/') {
      return obj;
    }
    
    // Remove leading slash
    const cleanPath = path.startsWith('/') ? path.slice(1) : path;
    const parts = cleanPath.split(/[\/\.]/);
    
    let current = obj;
    for (const part of parts) {
      // Handle array indices like items[0]
      const arrayMatch = part.match(/^(\w+)\[(\d+)\]$/);
      if (arrayMatch) {
        const [, key, index] = arrayMatch;
        current = current?.[key]?.[parseInt(index)];
      } else {
        current = current?.[part];
      }
      
      if (current === undefined) {
        return undefined;
      }
    }
    
    return current;
  }
  
  /**
   * Set nested value in object by path
   * @param {Object} obj - Object
   * @param {string} path - Path
   * @param {*} value - Value to set
   */
  setNestedValue(obj, path, value) {
    if (!path || path === '/') {
      Object.assign(obj, value);
      return;
    }
    
    const cleanPath = path.startsWith('/') ? path.slice(1) : path;
    const parts = cleanPath.split(/[\/\.]/);
    
    let current = obj;
    for (let i = 0; i < parts.length - 1; i++) {
      const part = parts[i];
      const arrayMatch = part.match(/^(\w+)\[(\d+)\]$/);
      
      if (arrayMatch) {
        const [, key, index] = arrayMatch;
        if (!current[key]) {
          current[key] = [];
        }
        if (!current[key][parseInt(index)]) {
          current[key][parseInt(index)] = {};
        }
        current = current[key][parseInt(index)];
      } else {
        if (!current[part]) {
          current[part] = {};
        }
        current = current[part];
      }
    }
    
    const lastPart = parts[parts.length - 1];
    const lastArrayMatch = lastPart.match(/^(\w+)\[(\d+)\]$/);
    if (lastArrayMatch) {
      const [, key, index] = lastArrayMatch;
      if (!current[key]) {
        current[key] = [];
      }
      current[key][parseInt(index)] = value;
    } else {
      current[lastPart] = value;
    }
  }
  
  /**
   * Delete nested value from object by path
   * @param {Object} obj - Object
   * @param {string} path - Path
   */
  deleteNestedValue(obj, path) {
    if (!path || path === '/') {
      return;
    }
    
    const cleanPath = path.startsWith('/') ? path.slice(1) : path;
    const parts = cleanPath.split(/[\/\.]/);
    
    let current = obj;
    for (let i = 0; i < parts.length - 1; i++) {
      const part = parts[i];
      const arrayMatch = part.match(/^(\w+)\[(\d+)\]$/);
      
      if (arrayMatch) {
        const [, key, index] = arrayMatch;
        current = current?.[key]?.[parseInt(index)];
      } else {
        current = current?.[part];
      }
      
      if (!current) {
        return;
      }
    }
    
    const lastPart = parts[parts.length - 1];
    const lastArrayMatch = lastPart.match(/^(\w+)\[(\d+)\]$/);
    if (lastArrayMatch) {
      const [, key, index] = lastArrayMatch;
      if (current[key] && Array.isArray(current[key])) {
        current[key].splice(parseInt(index), 1);
      }
    } else {
      delete current[lastPart];
    }
  }
  
  /**
   * Update data bindings after data model change
   * @param {string} surfaceId - Surface ID
   */
  updateDataBindings(surfaceId) {
    // Find all elements with data bindings and update them
    const surface = this.surfaces.get(surfaceId);
    if (!surface) {
      return;
    }
    
    // Re-render the surface
    // This is a simplified approach - in production, you'd want to update only affected components
    const elements = surface.element.querySelectorAll('[data-binding]');
    elements.forEach(element => {
      const path = element.getAttribute('data-binding');
      const value = this.getNestedValue(this.dataModels.get(surfaceId), path);
      if (value !== undefined) {
        if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
          element.value = value;
        } else {
          element.textContent = value;
        }
      }
    });
  }
  
  // Component renderers
  
  renderText(component, surfaceId) {
    const element = document.createElement('div');
    const text = this.resolveValue(component.text, surfaceId) || '';
    const usageHint = component.usageHint || 'body';
    
    // Apply usage hint as class
    element.className += ` a2ui-text-${usageHint}`;
    
    // Simple markdown support (basic)
    element.innerHTML = this.renderMarkdown(text);
    
    return element;
  }
  
  renderImage(component, surfaceId) {
    const element = document.createElement('img');
    const url = this.resolveValue(component.url, surfaceId) || '';
    element.src = url;
    element.alt = component.alt || '';
    
    if (component.fit) {
      element.style.objectFit = component.fit;
    }
    
    if (component.usageHint) {
      element.className += ` a2ui-image-${component.usageHint}`;
    }
    
    return element;
  }
  
  renderIcon(component, surfaceId) {
    const element = document.createElement('span');
    element.className = 'a2ui-icon material-icons';
    
    const iconName = typeof component.name === 'string' 
      ? component.name 
      : this.resolveValue(component.name?.path, surfaceId) || 'help';
    
    element.textContent = this.mapIconName(iconName);
    
    return element;
  }
  
  renderButton(component, componentMap, surfaceId) {
    const element = document.createElement('button');
    element.className = 'a2ui-button';
    
    if (component.primary) {
      element.className += ' a2ui-button-primary';
    }
    
    // Render child component
    if (component.child) {
      const childComponent = componentMap.get(component.child);
      if (childComponent) {
        const childElement = this.renderComponent(childComponent, componentMap, surfaceId);
        if (childElement) {
          element.appendChild(childElement);
        }
      }
    }
    
    // Handle action
    if (component.action) {
      element.addEventListener('click', () => {
        this.handleAction(component.action, surfaceId);
      });
    }
    
    return element;
  }
  
  renderCard(component, componentMap, surfaceId) {
    const element = document.createElement('div');
    element.className = 'a2ui-card';
    
    if (component.child) {
      const childComponent = componentMap.get(component.child);
      if (childComponent) {
        const childElement = this.renderComponent(childComponent, componentMap, surfaceId);
        if (childElement) {
          element.appendChild(childElement);
        }
      }
    }
    
    return element;
  }
  
  renderRow(component, componentMap, surfaceId) {
    const element = document.createElement('div');
    element.className = 'a2ui-row';
    element.style.display = 'flex';
    element.style.flexDirection = 'row';
    
    if (component.distribution) {
      element.style.justifyContent = this.mapDistribution(component.distribution);
    }
    
    if (component.alignment) {
      element.style.alignItems = component.alignment;
    }
    
    // Render children
    if (component.children) {
      const children = this.resolveChildren(component.children, componentMap, surfaceId);
      children.forEach(childId => {
        const childComponent = componentMap.get(childId);
        if (childComponent) {
          const childElement = this.renderComponent(childComponent, componentMap, surfaceId);
          if (childElement) {
            element.appendChild(childElement);
          }
        }
      });
    }
    
    return element;
  }
  
  renderColumn(component, componentMap, surfaceId) {
    const element = document.createElement('div');
    element.className = 'a2ui-column';
    element.style.display = 'flex';
    element.style.flexDirection = 'column';
    
    if (component.distribution) {
      element.style.justifyContent = this.mapDistribution(component.distribution);
    }
    
    if (component.alignment) {
      element.style.alignItems = component.alignment;
    }
    
    // Render children
    if (component.children) {
      const children = this.resolveChildren(component.children, componentMap, surfaceId);
      children.forEach(childId => {
        const childComponent = componentMap.get(childId);
        if (childComponent) {
          const childElement = this.renderComponent(childComponent, componentMap, surfaceId);
          if (childElement) {
            element.appendChild(childElement);
          }
        }
      });
    }
    
    return element;
  }
  
  renderList(component, componentMap, surfaceId) {
    const element = document.createElement('div');
    element.className = 'a2ui-list';
    element.style.display = 'flex';
    element.style.flexDirection = component.direction === 'horizontal' ? 'row' : 'column';
    
    if (component.alignment) {
      element.style.alignItems = component.alignment;
    }
    
    // Render children
    if (component.children) {
      const children = this.resolveChildren(component.children, componentMap, surfaceId);
      children.forEach(childId => {
        const childComponent = componentMap.get(childId);
        if (childComponent) {
          const childElement = this.renderComponent(childComponent, componentMap, surfaceId);
          if (childElement) {
            element.appendChild(childElement);
          }
        }
      });
    }
    
    return element;
  }
  
  renderTextField(component, surfaceId) {
    const element = document.createElement('input');
    element.type = this.mapTextFieldType(component.usageHint);
    element.className = 'a2ui-textfield';
    
    if (component.label) {
      const label = document.createElement('label');
      label.textContent = this.resolveValue(component.label, surfaceId);
      label.className = 'a2ui-label';
      
      const container = document.createElement('div');
      container.className = 'a2ui-textfield-container';
      container.appendChild(label);
      container.appendChild(element);
      
      if (component.text) {
        element.value = this.resolveValue(component.text, surfaceId) || '';
      }
      
      if (component.validationRegexp) {
        element.pattern = component.validationRegexp;
      }
      
      return container;
    }
    
    if (component.text) {
      element.value = this.resolveValue(component.text, surfaceId) || '';
    }
    
    return element;
  }
  
  renderCheckBox(component, surfaceId) {
    const element = document.createElement('input');
    element.type = 'checkbox';
    element.className = 'a2ui-checkbox';
    
    const label = document.createElement('label');
    label.textContent = this.resolveValue(component.label, surfaceId) || '';
    
    const value = this.resolveValue(component.value, surfaceId);
    element.checked = value === true;
    
    const container = document.createElement('div');
    container.className = 'a2ui-checkbox-container';
    container.appendChild(element);
    container.appendChild(label);
    
    return container;
  }
  
  renderSlider(component, surfaceId) {
    const element = document.createElement('input');
    element.type = 'range';
    element.className = 'a2ui-slider';
    element.min = component.min || 0;
    element.max = component.max || 100;
    element.value = this.resolveValue(component.value, surfaceId) || element.min;
    
    if (component.label) {
      const label = document.createElement('label');
      label.textContent = this.resolveValue(component.label, surfaceId);
      label.className = 'a2ui-label';
      
      const container = document.createElement('div');
      container.className = 'a2ui-slider-container';
      container.appendChild(label);
      container.appendChild(element);
      
      return container;
    }
    
    return element;
  }
  
  renderChoicePicker(component, componentMap, surfaceId) {
    const element = document.createElement('div');
    element.className = 'a2ui-choicepicker';
    
    if (component.label) {
      const label = document.createElement('label');
      label.textContent = this.resolveValue(component.label, surfaceId);
      label.className = 'a2ui-label';
      element.appendChild(label);
    }
    
    const selectedValues = this.resolveValue(component.value, surfaceId) || [];
    const isMultiple = component.usageHint === 'multipleSelection';
    
    component.options.forEach(option => {
      const optionElement = isMultiple 
        ? document.createElement('input')
        : document.createElement('input');
      
      optionElement.type = isMultiple ? 'checkbox' : 'radio';
      optionElement.name = component.id;
      optionElement.value = option.value;
      optionElement.checked = selectedValues.includes(option.value);
      
      const optionLabel = document.createElement('label');
      optionLabel.textContent = this.resolveValue(option.label, surfaceId);
      
      const optionContainer = document.createElement('div');
      optionContainer.className = 'a2ui-choice-option';
      optionContainer.appendChild(optionElement);
      optionContainer.appendChild(optionLabel);
      element.appendChild(optionContainer);
    });
    
    return element;
  }
  
  renderTabs(component, componentMap, surfaceId) {
    const element = document.createElement('div');
    element.className = 'a2ui-tabs';
    
    // Tab headers
    const tabHeaders = document.createElement('div');
    tabHeaders.className = 'a2ui-tab-headers';
    
    // Tab content
    const tabContent = document.createElement('div');
    tabContent.className = 'a2ui-tab-content';
    
    let activeTabIndex = 0;
    
    component.tabItems.forEach((tabItem, index) => {
      const header = document.createElement('button');
      header.className = 'a2ui-tab-header';
      header.textContent = this.resolveValue(tabItem.title, surfaceId);
      header.addEventListener('click', () => {
        // Update active tab
        tabHeaders.querySelectorAll('.a2ui-tab-header').forEach((h, i) => {
          h.classList.toggle('active', i === index);
        });
        tabContent.querySelectorAll('.a2ui-tab-panel').forEach((p, i) => {
          p.style.display = i === index ? 'block' : 'none';
        });
      });
      
      if (index === activeTabIndex) {
        header.classList.add('active');
      }
      
      tabHeaders.appendChild(header);
      
      // Tab panel
      const panel = document.createElement('div');
      panel.className = 'a2ui-tab-panel';
      panel.style.display = index === activeTabIndex ? 'block' : 'none';
      
      const childComponent = componentMap.get(tabItem.child);
      if (childComponent) {
        const childElement = this.renderComponent(childComponent, componentMap, surfaceId);
        if (childElement) {
          panel.appendChild(childElement);
        }
      }
      
      tabContent.appendChild(panel);
    });
    
    element.appendChild(tabHeaders);
    element.appendChild(tabContent);
    
    return element;
  }
  
  renderModal(component, componentMap, surfaceId) {
    const element = document.createElement('div');
    element.className = 'a2ui-modal';
    
    // Entry point (button that opens modal)
    const entryPoint = componentMap.get(component.entryPointChild);
    if (entryPoint) {
      const entryElement = this.renderComponent(entryPoint, componentMap, surfaceId);
      if (entryElement) {
        entryElement.addEventListener('click', () => {
          element.classList.add('active');
        });
        element.appendChild(entryElement);
      }
    }
    
    // Modal overlay
    const overlay = document.createElement('div');
    overlay.className = 'a2ui-modal-overlay';
    overlay.addEventListener('click', () => {
      element.classList.remove('active');
    });
    
    // Modal content
    const content = document.createElement('div');
    content.className = 'a2ui-modal-content';
    
    const contentChild = componentMap.get(component.contentChild);
    if (contentChild) {
      const contentElement = this.renderComponent(contentChild, componentMap, surfaceId);
      if (contentElement) {
        content.appendChild(contentElement);
      }
    }
    
    overlay.appendChild(content);
    element.appendChild(overlay);
    
    return element;
  }
  
  renderDivider(component) {
    const element = document.createElement('hr');
    element.className = 'a2ui-divider';
    if (component.axis === 'vertical') {
      element.style.width = '1px';
      element.style.height = '100%';
      element.style.borderLeft = '1px solid #ccc';
      element.style.borderTop = 'none';
    }
    return element;
  }
  
  renderVideo(component, surfaceId) {
    const element = document.createElement('video');
    element.src = this.resolveValue(component.url, surfaceId) || '';
    element.controls = true;
    element.className = 'a2ui-video';
    return element;
  }
  
  renderAudioPlayer(component, surfaceId) {
    const element = document.createElement('audio');
    element.src = this.resolveValue(component.url, surfaceId) || '';
    element.controls = true;
    element.className = 'a2ui-audioplayer';
    
    if (component.description) {
      const desc = document.createElement('div');
      desc.textContent = this.resolveValue(component.description, surfaceId);
      desc.className = 'a2ui-audio-description';
      
      const container = document.createElement('div');
      container.className = 'a2ui-audioplayer-container';
      container.appendChild(desc);
      container.appendChild(element);
      return container;
    }
    
    return element;
  }
  
  /**
   * Resolve children property (array or template)
   * @param {Array|Object} children - Children definition
   * @param {Map} componentMap - Component map
   * @param {string} surfaceId - Surface ID
   * @returns {Array<string>} - Array of child component IDs
   */
  resolveChildren(children, componentMap, surfaceId) {
    if (Array.isArray(children)) {
      return children;
    } else if (children && children.componentId && children.path) {
      // Template-based children
      const dataModel = this.dataModels.get(surfaceId);
      const dataList = this.getNestedValue(dataModel, children.path) || [];
      
      // For now, return the template component ID for each item
      // In a full implementation, you'd create component instances for each data item
      return dataList.map((_, index) => children.componentId);
    }
    return [];
  }
  
  /**
   * Handle action (button click, etc.)
   * @param {Object} action - Action definition
   * @param {string} surfaceId - Surface ID
   */
  handleAction(action, surfaceId) {
    console.log('Action triggered:', action);
    
    // Send action to Streamlit (if available)
    // Streamlit components.html uses iframe, so we need to communicate with parent
    if (window.parent && window.parent.postMessage) {
      window.parent.postMessage({
        type: 'a2ui-action',
        action: action.name,
        context: action.context,
        surfaceId: surfaceId,
        timestamp: new Date().toISOString()
      }, '*');
    }
    
    // Also try Streamlit's custom message format
    if (window.parent && window.parent.streamlit) {
      try {
        window.parent.streamlit.setComponentValue({
          type: 'a2ui-action',
          action: action.name,
          context: action.context,
          surfaceId: surfaceId
        });
      } catch (e) {
        // Streamlit API might not be available
        console.debug('Streamlit API not available:', e);
      }
    }
    
    // Also dispatch custom event for local handling
    if (window.dispatchEvent) {
      window.dispatchEvent(new CustomEvent('a2ui-action', {
        detail: {
          action: action.name,
          context: action.context,
          surfaceId: surfaceId
        }
      }));
    }
  }
  
  // Helper methods
  
  renderMarkdown(text) {
    // Simple markdown rendering (basic)
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
  }
  
  mapIconName(iconName) {
    // Map A2UI icon names to Material Icons
    const iconMap = {
      'accountCircle': 'account_circle',
      'add': 'add',
      'arrowBack': 'arrow_back',
      'arrowForward': 'arrow_forward',
      'attachFile': 'attach_file',
      'calendarToday': 'calendar_today',
      'call': 'call',
      'camera': 'camera_alt',
      'check': 'check',
      'close': 'close',
      'delete': 'delete',
      'download': 'download',
      'edit': 'edit',
      'event': 'event',
      'error': 'error',
      'favorite': 'favorite',
      'favoriteOff': 'favorite_border',
      'folder': 'folder',
      'help': 'help',
      'home': 'home',
      'info': 'info',
      'locationOn': 'location_on',
      'lock': 'lock',
      'lockOpen': 'lock_open',
      'mail': 'mail',
      'menu': 'menu',
      'moreVert': 'more_vert',
      'moreHoriz': 'more_horiz',
      'notifications': 'notifications',
      'notificationsOff': 'notifications_off',
      'pause': 'pause',
      'payment': 'payment',
      'person': 'person',
      'phone': 'phone',
      'photo': 'photo',
      'play': 'play_arrow',
      'print': 'print',
      'refresh': 'refresh',
      'search': 'search',
      'send': 'send',
      'settings': 'settings',
      'share': 'share',
      'shoppingCart': 'shopping_cart',
      'star': 'star',
      'starHalf': 'star_half',
      'starOff': 'star_border',
      'stop': 'stop',
      'upload': 'upload',
      'visibility': 'visibility',
      'visibilityOff': 'visibility_off',
      'volumeUp': 'volume_up',
      'volumeDown': 'volume_down',
      'volumeMute': 'volume_mute',
      'volumeOff': 'volume_off',
      'warning': 'warning'
    };
    
    return iconMap[iconName] || iconName;
  }
  
  mapDistribution(distribution) {
    const map = {
      'start': 'flex-start',
      'end': 'flex-end',
      'center': 'center',
      'spaceBetween': 'space-between',
      'spaceAround': 'space-around',
      'spaceEvenly': 'space-evenly',
      'stretch': 'stretch'
    };
    return map[distribution] || distribution;
  }
  
  mapTextFieldType(usageHint) {
    const map = {
      'shortText': 'text',
      'longText': 'textarea',
      'number': 'number',
      'obscured': 'password'
    };
    return map[usageHint] || 'text';
  }
}

// Export for use in Streamlit
if (typeof window !== 'undefined') {
  window.A2UIRenderer = A2UIRenderer;
}

