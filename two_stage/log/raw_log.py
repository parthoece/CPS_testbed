import csv
import os
import logging
from datetime import datetime
import re

class LogProcessor:
    def __init__(self, input_dir, output_file, log_file):
        
        self.input_dir = input_dir
        self.output_file = output_file
        self.log_file = log_file
        self.log_messages = []
        self.last_timestamp = None  

        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def process_logs(self):
        
        combined_data = []
        total_files = 0

        for root, _, files in os.walk(self.input_dir):
            for file in files:
                input_file = os.path.join(root, file)
                if file.endswith(('.log', '.csv', '.txt')):
                    total_files += 1
                    logging.info(f"Processing file: {input_file}")

                    if file.endswith('.csv'):
                        parsed_data = self._process_csv(input_file)
                    else:
                        parsed_data = self._process_text_log(input_file)

                    logging.info(f"Extracted {len(parsed_data)} logs from {input_file}")

                    combined_data.extend(parsed_data)
                    logging.info(f"Parsed {len(parsed_data)} logs from {file}")

        combined_data.sort(key=lambda x: x['Timestamp'])
        if combined_data:
            print(f"Extracted {len(combined_data)} logs.")
            print(f"Sample logs: {combined_data[:5]}")
        else:
            print("No logs were extracted. Check log format and parsing logic.")


        self._write_to_csv(combined_data)
        self._write_log_summary(total_files, len(combined_data))
        print(f"Total logs collected: {len(combined_data)}")
        


    def _process_text_log(self, input_file):
     
        parsed_data = []
        with open(input_file, 'r') as file:
            for line in file:
                timestamp, component, message = self._extract_log_data(line)
                logging.debug(f"Skipping unmatched log line: {line.strip()}")

                if not timestamp and self.last_timestamp:
                    timestamp = self.last_timestamp  

                if timestamp and component and message:
                    parsed_data.append({
                        "Timestamp": timestamp,
                        "Component": component,
                        "Message": message
                    })
                    self.last_timestamp = timestamp 

        return parsed_data

    def _extract_log_data(self, line):

        patterns = [
            r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) .*?\[(.*?)\] (.*)$',  # Standard log format
            r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(INFO|ERROR|DEBUG|WARN).*?\[(.*?)\] (.*)$',
            r'^(\w{3} \d{1,2} \d{2}:\d{2}:\d{2}) (.*)$'  # E.g., Feb 11 20:28:27 logs
        ]

        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                if len(match.groups()) == 3:
                    return self._parse_timestamp(match.group(1)), match.group(2), match.group(3)
                elif len(match.groups()) == 4:
                    return self._parse_timestamp(match.group(1)), match.group(3), match.group(4)

        return None, None, None

    def _process_csv(self, input_file):
      
        parsed_data = []
        with open(input_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                timestamp = self._parse_timestamp(row.get("Timestamp", ""))
                component = row.get("Component", "")
                message = row.get("Message", "")

                if not timestamp and self.last_timestamp:
                    timestamp = self.last_timestamp  

                if timestamp and component and message:
                    parsed_data.append({
                        "Timestamp": timestamp,
                        "Component": component,
                        "Message": message
                    })
                    self.last_timestamp = timestamp  

        return parsed_data

    def _parse_timestamp(self, raw_timestamp):
      
        try:
            for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S", "%b %d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
                try:
                    return datetime.strptime(raw_timestamp, fmt).isoformat()
                except ValueError:
                    continue
        except Exception as e:
            logging.error(f"Error parsing timestamp '{raw_timestamp}': {e}")
        return None 

    def _write_to_csv(self, data):
       
        if not data:
            logging.warning("No logs to write. Skipping CSV write.")
            print("No logs to write. CSV file will be empty.")
            return  
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["Timestamp", "Component", "Message"])
            writer.writeheader()
            writer.writerows(data)

        logging.info(f"Combined logs saved to {self.output_file}")
        print(f"Successfully wrote {len(data)} logs to CSV.")


    def _write_log_summary(self, total_files, total_logs):
       
        logging.info(f"Total files processed: {total_files}")
        logging.info(f"Total logs written to CSV: {total_logs}")



if __name__ == "__main__":
    input_dir = "/home/smasher/Downloads/cps_tesbed/data/attack_data/syslog"
    output_dir = "/home/smasher/Downloads/cps_tesbed/ML/preprocess/uldata/log"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "attack_combined_logs.csv")
    log_file = os.path.join(output_dir, "log_processing_info.log")

    processor = LogProcessor(input_dir, output_file, log_file)
    processor.process_logs()
