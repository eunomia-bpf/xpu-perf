export class DataExporter {
  static exportToCSV(data: any[], filename: string = 'profiler_data.csv'): void {
    const csvContent = [
      'Function Name,Self Time,Total Time,Call Count,Thread',
      ...data.map(row => 
        `"${row.name}","${row.selfTime}","${row.totalTime}",${row.callCount},"${row.thread}"`
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  static exportToJSON(data: any, filename: string = 'profiler_data.json'): void {
    const jsonContent = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
  }
} 