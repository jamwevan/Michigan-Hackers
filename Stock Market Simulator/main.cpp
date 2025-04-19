#include <iostream>
#include <vector>
#include <queue>
#include <sstream>
#include <algorithm>
#include <getopt.h>
#include "P2random.h"
#include <limits>

// TimeTraveler struct to track buy and sell states
struct TimeTraveler {
    unsigned int bestSellTime;
    unsigned int bestSellPrice;
    unsigned int bestBuyTime;
    unsigned int bestBuyPrice;

    unsigned int potentialBuyTime;
    unsigned int potentialBuyPrice;

    char state; // 'n', 'b', 'c', or 'p'

    TimeTraveler()
        : bestSellTime(0), bestSellPrice(0), bestBuyTime(0), bestBuyPrice(0),
          potentialBuyTime(0), potentialBuyPrice(0), state('n') {}
};

// Order struct to represent buy/sell orders
struct Order {
    unsigned int timestamp;
    unsigned int trader_id;
    unsigned int stock_id;
    bool is_buy;
    unsigned int price;
    unsigned int quantity;
    int unique_id;  // New field
    
    Order(unsigned int t, unsigned int trader, unsigned int stock, bool buy, unsigned int p, unsigned int qty, int id)
        : timestamp(t), trader_id(trader), stock_id(stock), is_buy(buy), price(p), quantity(qty), unique_id(id) {}
};

// Comparator for Buy Orders (Max-Heap based on price, then min timestamp)
struct BuyOrderComparator {
    bool operator()(const Order& a, const Order& b) const {
        if (a.price != b.price) return a.price < b.price; // Higher price has higher priority
        if (a.timestamp != b.timestamp) return a.timestamp > b.timestamp; // Earlier timestamp has higher priority
        return a.unique_id > b.unique_id; // Lower unique_id (earlier order) has higher priority
    }
};

struct SellOrderComparator {
    bool operator()(const Order& a, const Order& b) const {
        if (a.price != b.price) return a.price > b.price; // Lower price has higher priority
        if (a.timestamp != b.timestamp) return a.timestamp > b.timestamp; // Earlier timestamp has higher priority
        return a.unique_id > b.unique_id; // Lower unique_id (earlier order) has higher priority
    }
};

class Market {
private:
    // Priority queues for each stock
    std::vector<std::priority_queue<Order, std::vector<Order>, BuyOrderComparator>> buy_orders;
    std::vector<std::priority_queue<Order, std::vector<Order>, SellOrderComparator>> sell_orders;
    
    // Time Travelers for each stock
    std::vector<TimeTraveler> time_travelers;
    
    // Current timestamp
    unsigned int current_timestamp;
    
    // Total trades completed
    unsigned int total_trades;
    
    // Trader statistics
    std::vector<unsigned int> shares_bought;
    std::vector<unsigned int> shares_sold;
    std::vector<long long> net_transfer;
    
    // Median tracking
    std::vector<std::vector<unsigned int>> trade_prices;
    
    // Command-line options
    bool median_mode;
    bool verbose_mode;
    bool time_travelers_mode;
    bool trader_info_mode;
    
    // Number of traders and stocks
    size_t num_traders;
    size_t num_stocks;
    int order_counter;

public:
    Market()
        : current_timestamp(0), total_trades(0), median_mode(false),
          verbose_mode(false), time_travelers_mode(false), trader_info_mode(false),
          num_traders(0), num_stocks(0), order_counter(0) {}

    void initialize(size_t num_traders, size_t num_stocks) {
        this->num_traders = num_traders;
        this->num_stocks = num_stocks;
        buy_orders.resize(num_stocks);
        sell_orders.resize(num_stocks);
        time_travelers.resize(num_stocks);
        shares_bought.resize(num_traders, 0);
        shares_sold.resize(num_traders, 0);
        net_transfer.resize(num_traders, 0);
        if (median_mode) {
            trade_prices.resize(num_stocks);
        }
    }

    void set_modes(bool median, bool verbose, bool time_travelers, bool trader_info) {
        median_mode = median;
        verbose_mode = verbose;
        time_travelers_mode = time_travelers;
        trader_info_mode = trader_info;
        if (median_mode) {
            trade_prices.resize(num_stocks);
        }
    }

    void processOrder(const Order& order) {
        // Check for non-decreasing timestamp
        if (order.timestamp < current_timestamp) {
            std::cerr << "Error: Timestamps must be non-decreasing\n";
            exit(1);
        }

        // If timestamp changes, handle median output
        if (order.timestamp != current_timestamp) {
            if (median_mode) {
                outputMedian();
            }
            current_timestamp = order.timestamp;
        }

        Order new_order(order.timestamp, order.trader_id, order.stock_id, order.is_buy, order.price, order.quantity, order_counter++);

        // Match the order
        if (new_order.is_buy) {
            matchBuyOrder(new_order);
        } else {
            matchSellOrder(new_order);
        }

        // Update time traveler information
        if (time_travelers_mode) {
            updateTimeTraveler(order);
        }
    }

    void matchBuyOrder(Order buy_order) {
        auto& sell_queue = sell_orders[buy_order.stock_id];

        while (!sell_queue.empty() && buy_order.quantity > 0 &&
               sell_queue.top().price <= buy_order.price) {
            Order sell_order = sell_queue.top();
            sell_queue.pop();

            unsigned int trade_price = (sell_order.timestamp <= buy_order.timestamp) ? sell_order.price : buy_order.price;
            unsigned int trade_qty = std::min(buy_order.quantity, sell_order.quantity);

            processTrade(buy_order, sell_order, trade_price, trade_qty);

            buy_order.quantity -= trade_qty;
            sell_order.quantity -= trade_qty;

            if (sell_order.quantity > 0) {
                sell_queue.push(sell_order);
            }
        }

        if (buy_order.quantity > 0) {
            buy_orders[buy_order.stock_id].push(buy_order);
        }
    }

    void matchSellOrder(Order sell_order) {
        auto& buy_queue = buy_orders[sell_order.stock_id];

        while (!buy_queue.empty() && sell_order.quantity > 0 &&
               buy_queue.top().price >= sell_order.price) {
            Order buy_order = buy_queue.top();
            buy_queue.pop();

            unsigned int trade_price = (buy_order.timestamp <= sell_order.timestamp) ? buy_order.price : sell_order.price;
            unsigned int trade_qty = std::min(buy_order.quantity, sell_order.quantity);

            processTrade(buy_order, sell_order, trade_price, trade_qty);

            sell_order.quantity -= trade_qty;
            buy_order.quantity -= trade_qty;

            if (buy_order.quantity > 0) {
                buy_queue.push(buy_order);
            }
        }

        if (sell_order.quantity > 0) {
            sell_orders[sell_order.stock_id].push(sell_order);
        }
    }

    void processTrade(const Order& buyer, const Order& seller, unsigned int price, unsigned int quantity) {
        if (verbose_mode) {
            std::cout << "Trader " << buyer.trader_id << " purchased " << quantity
                      << " shares of Stock " << buyer.stock_id << " from Trader "
                      << seller.trader_id << " for $" << price << "/share\n";
        }

        shares_bought[buyer.trader_id] += quantity;
        shares_sold[seller.trader_id] += quantity;
        net_transfer[buyer.trader_id] -= static_cast<long long>(price) * quantity;
        net_transfer[seller.trader_id] += static_cast<long long>(price) * quantity;

        total_trades++;

        // Record trade price for median calculation
        if (median_mode) {
            trade_prices[buyer.stock_id].push_back(price);
        }
    }

    void outputMedian() {
        for (size_t i = 0; i < num_stocks; ++i) {
            if (!trade_prices[i].empty()) {
                size_t size = trade_prices[i].size();
                size_t mid = size / 2;

                // Sort the prices to find the median
                std::vector<unsigned int> temp = trade_prices[i];
                std::sort(temp.begin(), temp.end());

                unsigned int median = temp[mid];
                if (size % 2 == 0) {
                    median = (median + temp[mid - 1]) / 2; // Use integer division
                }

                std::cout << "Median match price of Stock " << i
                          << " at time " << current_timestamp
                          << " is $" << median << "\n";
            }
        }
    }

    void traderInfoOutput() {
        if (trader_info_mode) {
            std::cout << "---Trader Info---\n";
            for (size_t i = 0; i < num_traders; ++i) {
                std::cout << "Trader " << i << " bought " << shares_bought[i]
                          << " and sold " << shares_sold[i]
                          << " for a net transfer of $" << net_transfer[i] << "\n";
            }
        }
    }

    void updateTimeTraveler(const Order& o) {
        TimeTraveler& traveler = time_travelers[o.stock_id];

        if (traveler.state == 'n') {
            if (!o.is_buy) { // 'S' corresponds to sell orders
                traveler.bestBuyPrice = o.price;
                traveler.bestBuyTime = o.timestamp;
                traveler.state = 'b';
            }
        }
        else if (traveler.state == 'b') {
            if (!o.is_buy && o.price < traveler.bestBuyPrice) {
                traveler.bestBuyPrice = o.price;
                traveler.bestBuyTime = o.timestamp;
            }
            else if (o.is_buy && o.price > traveler.bestBuyPrice) {
                traveler.bestSellPrice = o.price;
                traveler.bestSellTime = o.timestamp;
                traveler.state = 'c';
            }
            else if (!o.is_buy && o.price < traveler.bestBuyPrice) {
                traveler.potentialBuyPrice = o.price;
                traveler.potentialBuyTime = o.timestamp;
                traveler.state = 'p';
            }
        }
        else if (traveler.state == 'c') {
            if (o.is_buy && o.price > traveler.bestSellPrice) {
                traveler.bestSellPrice = o.price;
                traveler.bestSellTime = o.timestamp;
            }
            else if (!o.is_buy && o.price < traveler.bestBuyPrice) {
                traveler.potentialBuyPrice = o.price;
                traveler.potentialBuyTime = o.timestamp;
                traveler.state = 'p';
            }
        }
        else if (traveler.state == 'p') {
            if (o.is_buy &&
                static_cast<int>(o.price) - static_cast<int>(traveler.potentialBuyPrice) >
                static_cast<int>(traveler.bestSellPrice) - static_cast<int>(traveler.bestBuyPrice)) {
                traveler.bestBuyPrice = traveler.potentialBuyPrice;
                traveler.bestSellPrice = o.price;
                traveler.bestBuyTime = traveler.potentialBuyTime;
                traveler.bestSellTime = o.timestamp;
                traveler.state = 'c';
            }
        }
    }

    void timeTravelersOutput() const {
        if (time_travelers_mode) {
            std::cout << "---Time Travelers---\n";
            for (size_t stock_id = 0; stock_id < num_stocks; ++stock_id) {
                const TimeTraveler& traveler = time_travelers[stock_id];
                if (traveler.bestSellPrice > traveler.bestBuyPrice) {
                    std::cout << "A time traveler would buy Stock " << stock_id
                              << " at time " << traveler.bestBuyTime << " for $" << traveler.bestBuyPrice
                              << " and sell it at time " << traveler.bestSellTime
                              << " for $" << traveler.bestSellPrice << "\n";
                } else {
                    std::cout << "A time traveler could not make a profit on Stock " << stock_id << "\n";
                }
            }
        }
    }

    void processOrders(std::istream& input_stream) {
        unsigned int timestamp, trader_id, stock_id, price, quantity;
        bool is_buy;
        std::string intent;
        char junk;
        int order_counter = 0;  // Initialize the order counter

        while (input_stream >> timestamp >> intent >> junk >> trader_id >> junk
               >> stock_id >> junk >> price >> junk >> quantity) {
            is_buy = (intent == "BUY");
            Order new_order(timestamp, trader_id, stock_id, is_buy, price, quantity, order_counter++);
            processOrder(new_order);
        }
    }

    void summaryOutput() {
        std::cout << "---End of Day---\n";
        std::cout << "Trades Completed: " << total_trades << "\n";
    }
};

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    bool median_mode = false, verbose_mode = false,
         trader_info_mode = false, time_travelers_mode = false;

    int option_char;
    static struct option long_options[] = {
        {"verbose", no_argument, 0, 'v'},
        {"median", no_argument, 0, 'm'},
        {"trader_info", no_argument, 0, 'i'},
        {"time_travelers", no_argument, 0, 't'},
        {0, 0, 0, 0}
    };

    while ((option_char = getopt_long(argc, argv, "vmit", long_options, NULL)) != -1) {
        switch (option_char) {
            case 'v': verbose_mode = true; break;
            case 'm': median_mode = true; break;
            case 'i': trader_info_mode = true; break;
            case 't': time_travelers_mode = true; break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-v] [-m] [-i] [-t]\n";
                exit(1);
        }
    }

    Market market;
    int order_counter = 0;

    std::string line, mode;
    std::getline(std::cin, line); // Comment line
    std::getline(std::cin, line); // Mode line
    mode = line.substr(line.find(":") + 1);
    mode = mode.substr(mode.find_first_not_of(" \t"));

    std::getline(std::cin, line); // NUM_TRADERS line
    int temp_num_traders = std::stoi(line.substr(line.find(":") + 1));
    if (temp_num_traders < 0) {
        std::cerr << "Error: Number of traders cannot be negative.\n";
        exit(1);
    }
    size_t num_traders = static_cast<size_t>(temp_num_traders);

    std::getline(std::cin, line); // NUM_STOCKS line
    int temp_num_stocks = std::stoi(line.substr(line.find(":") + 1));
    if (temp_num_stocks < 0) {
        std::cerr << "Error: Number of stocks cannot be negative.\n";
        exit(1);
    }
    size_t num_stocks = static_cast<size_t>(temp_num_stocks);

    // Set modes before initialization
    market.set_modes(median_mode, verbose_mode, time_travelers_mode, trader_info_mode);

    // Initialize after setting modes
    market.initialize(num_traders, num_stocks);

    // Print "Processing orders..." before any other output
    std::cout << "Processing orders...\n";

    if (mode == "PR") {
        std::getline(std::cin, line); // RANDOM_SEED line
        unsigned long temp_seed = std::stoul(line.substr(line.find(":") + 1));
        if (temp_seed > std::numeric_limits<unsigned int>::max()) {
            std::cerr << "Error: Seed value out of range for unsigned int.\n";
            exit(1);
        }
        unsigned int seed = static_cast<unsigned int>(temp_seed);

        std::getline(std::cin, line); // NUMBER_OF_ORDERS line
        unsigned long temp_num_orders = std::stoul(line.substr(line.find(":") + 1));
        if (temp_num_orders > std::numeric_limits<unsigned int>::max()) {
            std::cerr << "Error: Number of orders out of range for unsigned int.\n";
            exit(1);
        }
        unsigned int num_orders = static_cast<unsigned int>(temp_num_orders);

        std::getline(std::cin, line); // ARRIVAL_RATE line
        unsigned long temp_arrival_rate = std::stoul(line.substr(line.find(":") + 1));
        if (temp_arrival_rate > std::numeric_limits<unsigned int>::max()) {
            std::cerr << "Error: Arrival rate out of range for unsigned int.\n";
            exit(1);
        }
        unsigned int arrival_rate = static_cast<unsigned int>(temp_arrival_rate);

        std::stringstream ss;
        P2random::PR_init(ss, seed, static_cast<unsigned int>(num_traders), static_cast<unsigned int>(num_stocks), num_orders, arrival_rate);
        market.processOrders(ss);
    } else if (mode == "TL") {
        // Read and process orders line by line
        int temp_timestamp, temp_trader_id, temp_stock_id, temp_price, temp_quantity;
        std::string intent;
        char junk;

        while (std::cin >> temp_timestamp >> intent >> junk >> temp_trader_id >> junk
               >> temp_stock_id >> junk >> temp_price >> junk >> temp_quantity) {
            // Input validation
            if (temp_timestamp < 0) {
                std::cerr << "Error: Negative timestamp encountered.\n";
                exit(1);
            }
            if (temp_trader_id < 0 || static_cast<size_t>(temp_trader_id) >= num_traders) {
                std::cerr << "Error: Trader ID " << temp_trader_id << " out of range.\n";
                exit(1);
            }
            if (temp_stock_id < 0 || static_cast<size_t>(temp_stock_id) >= num_stocks) {
                std::cerr << "Error: Stock ID " << temp_stock_id << " out of range.\n";
                exit(1);
            }
            if (temp_price <= 0) {
                std::cerr << "Error: Non-positive price encountered.\n";
                exit(1);
            }
            if (temp_quantity <= 0) {
                std::cerr << "Error: Non-positive quantity encountered.\n";
                exit(1);
            }

            bool is_buy = (intent == "BUY");
            Order new_order(static_cast<unsigned int>(temp_timestamp),
                           static_cast<unsigned int>(temp_trader_id),
                           static_cast<unsigned int>(temp_stock_id),
                           is_buy,
                           static_cast<unsigned int>(temp_price),
                           static_cast<unsigned int>(temp_quantity),
                           order_counter++);  // Add the unique ID
            market.processOrder(new_order);
        }
    } else {
        std::cerr << "Error: Invalid mode " << mode << "\n";
        return 1;
    }

    // If mode is TL and median is enabled, handle final median
    if (mode == "TL" && median_mode) {
        market.outputMedian();
    }
    
    if (mode == "PR" && median_mode) {
        market.outputMedian();
    }
    
    

    market.summaryOutput();
    market.traderInfoOutput();
    market.timeTravelersOutput();

    return 0;
}
