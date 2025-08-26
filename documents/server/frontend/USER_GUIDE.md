# 📖 User Guide: Data-Focused Profiler Interface

## 🎯 **Overview**

The profiler now uses a **data-focused approach** where you:
1. **Create analyzer instances** to generate data
2. **Select and combine data sources** from multiple analyzers  
3. **Choose views** that are compatible with your data

## 🚀 **Getting Started**

### **Step 1: Create Analyzer Instances**

1. In the **"Analyzer"** section (top-left):
   - Select an analyzer type (🔥 Flame Graph, ⏰ Wall Clock, 💤 Off-CPU)
   - Click **"+ New"** to create an instance
   - Give it a descriptive name (e.g., "Main Thread Profile")
   - Configure settings (duration, frequency, target process)
   - Click **"Create"**

2. **Run the analyzer**:
   - Select your instance from the dropdown
   - Click **"Start Analysis"**
   - Wait for completion (status shows "Completed")

### **Step 2: Select Data Sources**

1. In the **"Data Sources"** section (middle-left):
   - Click **"+ Select Data"** to create a data selection
   - Choose a name (e.g., "Combined CPU Profile")
   - Select one or more data sources:
     - ✅ Completed analyzer instances
     - ✅ Uploaded files (future)
     - ✅ API data (future)

3. **Choose combination mode**:
   - **Merge**: Combine object properties
   - **Append**: Join arrays together  
   - **Override**: Last source wins

4. Click **"Create"** - this becomes your active data

### **Step 3: Choose Compatible Views**

1. In the **"View Type"** section (bottom-left):
   - Views automatically show compatibility with your current data
   - ✅ **Compatible** views are enabled
   - ⚠️ **Incompatible** views show format warnings

2. **Available views**:
   - **🎯 3D Flame Graph**: Best for flamegraph data format
   - **📊 Data View**: Works with any data format (text display)

## 🔄 **Working with Multiple Data Sources**

### **Example: Combining Two Profiling Runs**

1. **Create multiple analyzer instances**:
   ```
   Instance A: "CPU Profile #1" (flamegraph format)
   Instance B: "CPU Profile #2" (flamegraph format)  
   ```

2. **Run both analyzers** and wait for completion

3. **Create combined data selection**:
   - Name: "Merged CPU Data"
   - Sources: [A, B]
   - Mode: "Merge"
   - Result: Combined flamegraph data

4. **Views automatically adapt**:
   - 3D Flame Graph: ✅ Shows merged data
   - Data View: ✅ Shows combined raw data

### **Example: Mixed Data Types**

1. **Create different analyzer types**:
   ```
   Instance A: Flame Graph (format: "flamegraph")
   Instance B: Wall Clock (format: "timeline")
   ```

2. **Combine with different modes**:
   - **Merge mode**: Creates "mixed" format
   - **Override mode**: Uses format of last source

3. **View compatibility**:
   - 3D Flame Graph: ⚠️ Mixed format warning, still displays
   - Data View: ✅ Always compatible

## 🎛️ **Interface Sections**

### **📊 Data Sources Panel**
- **Current Data**: Shows active data selection
- **Available Data Selections**: List of saved combinations
- **+ Select Data**: Create new data combinations

### **🔬 Analyzer Panel** 
- **Analyzer Type**: Choose profiler type
- **+ New**: Create analyzer instance
- **Instance dropdown**: Select/manage instances
- **Start/Stop**: Control analyzer execution

### **🎯 View Type Panel**
- **Radio buttons**: Select view type
- **Compatibility info**: Shows data format requirements
- **Format indicators**: Current data format and fields

### **📈 Main Viewport**
- **Dynamic content**: Changes based on selected view
- **Compatibility status**: Shows if data format is optimal
- **View-specific controls**: Configuration for current view

## 🔧 **Configuration Options**

### **Analyzer Configuration**
Each analyzer type has different settings:

- **🔥 Flame Graph Profiler**:
  - Duration (1-300 seconds)
  - Frequency (1-999 Hz)
  - Target Process (name or PID)

- **⏰ Wall Clock Analyzer**:
  - Duration, Frequency, Target Process
  - Include Off-CPU (boolean)

- **💤 Off-CPU Time Analyzer**:
  - Duration, Target Process
  - Min Block Time (microseconds)

### **View Configuration**
Views have minimal, focused settings:

- **🎯 3D Flame Graph**:
  - Color Scheme (Hot/Cold, Thread-based, Function-based)

- **📊 Data View**:
  - No configuration (pure data display)

## 📝 **Data Format Compatibility**

### **Format Types**
- **`flamegraph`**: Stack trace sampling data
- **`timeline`**: Time-series event data  
- **`sample`**: Generic sample data
- **`mixed`**: Combined multiple formats
- **`any`**: Accepts all formats (Data View only)

### **Compatibility Matrix**
| View | flamegraph | timeline | sample | mixed | any |
|------|------------|----------|--------|-------|-----|
| 3D Flame Graph | ✅ Optimal | ⚠️ Works | ⚠️ Works | ⚠️ Works | ❌ |
| Data View | ✅ | ✅ | ✅ | ✅ | ✅ |

## 💡 **Tips & Best Practices**

### **Naming Conventions**
- **Analyzer instances**: Descriptive names like "Main Thread 30s", "Worker Pool Profile"
- **Data selections**: Purpose-based like "Combined CPU Data", "Before/After Comparison"

### **Performance Considerations**
- **Large data sets**: Use "Override" mode instead of "Merge" for better performance
- **Many sources**: Limit to 3-5 sources per selection for UI responsiveness

### **Workflow Recommendations**
1. **Start simple**: Create single analyzer instances first
2. **Test compatibility**: Check how views display your data
3. **Combine strategically**: Only combine related data sources
4. **Save selections**: Reuse common data combinations

## 🐛 **Troubleshooting**

### **No Data Sources Available**
- **Cause**: No analyzer instances have completed
- **Solution**: Run at least one analyzer to completion

### **View Shows Format Warning**
- **Cause**: Current data format not optimal for selected view
- **Impact**: View still works but may not be ideal
- **Solution**: Try Data View for universal compatibility

### **Data Selection Creation Fails**
- **Cause**: No sources selected or invalid combination
- **Solution**: Select at least one completed data source

### **Views Not Updating**
- **Cause**: Data selection not properly activated
- **Solution**: Click on data selection in the list to activate it

## 🔮 **Future Features**

Coming soon:
- **File upload**: Direct data file import
- **API connections**: Real-time data streaming
- **More view types**: Timeline view, statistics view
- **Advanced filtering**: Filter data sources by criteria
- **Export options**: Save data selections and configurations

---

This data-focused approach gives you complete flexibility to mix and match profiling data from multiple sources while maintaining clear separation between data and visualization! 