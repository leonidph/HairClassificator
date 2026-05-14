#ifndef _CSV_HPP_
#define _CSV_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class CSVReader {
private:
    std::vector<std::vector<std::string>> data;

    // Helper function to trim whitespace
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, (last - first + 1));
    }

public:
    CSVReader() {}
    virtual ~CSVReader() {}
    // Read CSV file into vector
    bool readFile(const std::string& filename) {
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string field;
            std::vector<std::string> row;

            while (std::getline(ss, field, ',')) {
                row.push_back(trim(field));
            }

            if (!row.empty()) {
                data.push_back(row);
            }
        }

        file.close();
        return true;
    }

    // Get all data
    const std::vector<std::vector<std::string>>& getData() const {
        return data;
    }

    // Get number of rows
    size_t getRowCount() const {
        return data.size();
    }

    // Get number of columns in a row
    size_t getColumnCount(size_t row) const {
        if (row < data.size()) {
            return data[row].size();
        }
        return 0;
    }

    // Get a specific cell value
    std::string getCell(size_t row, size_t col) const {
        if (row < data.size() && col < data[row].size()) {
            return data[row][col];
        }
        return "";
    }

    // Print all data to console
    void printData() const {
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                std::cout << data[i][j];
                if (j < data[i].size() - 1) std::cout << " | ";
            }
            std::cout << std::endl;
        }
    }
 int getCellAsInt(size_t row, size_t col) const {
        std::string value = getCell(row, col);
        if (value.empty()) {
            std::cerr << "Error: Empty cell at row " << row << ", col " << col << std::endl;
            return 0;
        }
        try {
            return std::stoi(value);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: Cannot convert '" << value << "' to int at row " << row << ", col " << col << std::endl;
            return 0;
        } catch (const std::out_of_range& e) {
            std::cerr << "Error: Value '" << value << "' is out of range at row " << row << ", col " << col << std::endl;
            return 0;
        }
    }
    // Clear data
    void clear() {
        data.clear();
    }
};
#endif