import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_DEFAULT_KEY;

// Check if the key looks like a mock key (Supabase keys are JWTs starting with eyJ)
const isMockKey = !supabaseKey || !supabaseKey.startsWith('eyJ');

class MockSupabaseClient {
    constructor() {
        this._subscribers = [];
        this.auth = {
            getSession: async () => {
                try {
                    const sessionStr = localStorage.getItem('mock_supabase_session');
                    const session = sessionStr ? JSON.parse(sessionStr) : null;
                    return { data: { session }, error: null };
                } catch (e) {
                    return { data: { session: null }, error: null };
                }
            },
            getUser: async () => {
                try {
                    const sessionStr = localStorage.getItem('mock_supabase_session');
                    const session = sessionStr ? JSON.parse(sessionStr) : null;
                    return { data: { user: session ? session.user : null }, error: null };
                } catch (e) {
                    return { data: { user: null }, error: null };
                }
            },
            onAuthStateChange: (callback) => {
                try {
                    const sessionStr = localStorage.getItem('mock_supabase_session');
                    const session = sessionStr ? JSON.parse(sessionStr) : null;
                    setTimeout(() => {
                        callback('INITIAL_SESSION', session);
                    }, 0);
                } catch (e) {}
                
                this._subscribers.push(callback);
                
                return {
                    data: {
                        subscription: {
                            unsubscribe: () => {
                                this._subscribers = this._subscribers.filter(s => s !== callback);
                            }
                        }
                    }
                };
            },
            signInWithPassword: async ({ email, password }) => {
                const session = {
                    user: {
                        id: '00000000-0000-0000-0000-000000000000',
                        email: email || 'guest@crowdshield.org',
                    },
                    access_token: 'mock-access-token',
                };
                try {
                    localStorage.setItem('mock_supabase_session', JSON.stringify(session));
                } catch (e) {}
                
                this._subscribers.forEach(s => {
                    try { s('SIGNED_IN', session); } catch (e) {}
                });
                return { data: { session }, error: null };
            },
            signUp: async ({ email, password }) => {
                const session = {
                    user: {
                        id: '00000000-0000-0000-0000-000000000000',
                        email: email || 'guest@crowdshield.org',
                    },
                    access_token: 'mock-access-token',
                };
                try {
                    localStorage.setItem('mock_supabase_session', JSON.stringify(session));
                } catch (e) {}
                
                this._subscribers.forEach(s => {
                    try { s('SIGNED_IN', session); } catch (e) {}
                });
                return { data: { session }, error: null };
            },
            signOut: async () => {
                try {
                    localStorage.removeItem('mock_supabase_session');
                } catch (e) {}
                
                this._subscribers.forEach(s => {
                    try { s('SIGNED_OUT', null); } catch (e) {}
                });
                return { error: null };
            }
        };
    }

    from(table) {
        return new MockQueryBuilder(table);
    }

    channel(name) {
        return {
            on: (event, filter, callback) => {
                return {
                    subscribe: () => {
                        console.log(`[MockSupabase] Subscribed to real-time channel: ${name}`);
                        return { unsubscribe: () => {} };
                    }
                };
            },
            subscribe: () => {
                console.log(`[MockSupabase] Subscribed to channel: ${name}`);
                return { unsubscribe: () => {} };
            }
        };
    }

    removeChannel(channel) {
        // No-op
    }
}

class MockQueryBuilder {
    constructor(table) {
        this.table = table;
        this.filters = [];
        this.orderCol = null;
        this.orderOptions = null;
        this.operation = 'select';
    }

    _getData() {
        try {
            const key = `mock_table_${this.table}`;
            const dataStr = localStorage.getItem(key);
            return dataStr ? JSON.parse(dataStr) : [];
        } catch (e) {
            return [];
        }
    }

    _setData(data) {
        try {
            const key = `mock_table_${this.table}`;
            localStorage.setItem(key, JSON.stringify(data));
        } catch (e) {}
    }

    select(columns = '*') {
        this.operation = 'select';
        return this;
    }

    insert(records) {
        this.operation = 'insert';
        this.recordsToInsert = Array.isArray(records) ? records : [records];
        return this;
    }

    update(fields) {
        this.operation = 'update';
        this.fieldsToUpdate = fields;
        return this;
    }

    delete() {
        this.operation = 'delete';
        return this;
    }

    eq(column, value) {
        this.filters.push({ type: 'eq', column, value });
        return this;
    }

    order(column, options = {}) {
        this.orderCol = column;
        this.orderOptions = options;
        return this;
    }

    async then(onfulfilled, onrejected) {
        try {
            const res = await this.execute();
            return onfulfilled ? onfulfilled(res) : res;
        } catch (err) {
            if (onrejected) return onrejected(err);
            throw err;
        }
    }

    async execute() {
        let data = this._getData();

        if (this.operation === 'select') {
            // Apply filters
            for (const filter of this.filters) {
                if (filter.type === 'eq') {
                    data = data.filter(row => row[filter.column] === filter.value);
                }
            }
            if (this.orderCol) {
                const ascending = this.orderOptions?.ascending !== false;
                data.sort((a, b) => {
                    if (a[this.orderCol] < b[this.orderCol]) return ascending ? -1 : 1;
                    if (a[this.orderCol] > b[this.orderCol]) return ascending ? 1 : -1;
                    return 0;
                });
            }
            return { data, error: null };
        }

        if (this.operation === 'insert') {
            const newRecords = this.recordsToInsert.map(record => ({
                id: Math.random().toString(36).substring(2, 11),
                created_at: new Date().toISOString(),
                ...record
            }));
            data = [...data, ...newRecords];
            this._setData(data);
            return { data: newRecords[0], error: null };
        }

        if (this.operation === 'update') {
            let updatedRecords = [];
            data = data.map(row => {
                let matches = true;
                for (const filter of this.filters) {
                    if (filter.type === 'eq' && row[filter.column] !== filter.value) {
                        matches = false;
                    }
                }
                if (matches) {
                    const updated = { ...row, ...this.fieldsToUpdate };
                    updatedRecords.push(updated);
                    return updated;
                }
                return row;
            });
            this._setData(data);
            return { data: updatedRecords, error: null };
        }

        if (this.operation === 'delete') {
            data = data.filter(row => {
                let matches = true;
                for (const filter of this.filters) {
                    if (filter.type === 'eq' && row[filter.column] !== filter.value) {
                        matches = false;
                    }
                }
                return !matches;
            });
            this._setData(data);
            return { data: null, error: null };
        }

        return { data: null, error: null };
    }
}

// Intercept client initialization if mock key is detected
let supabaseClient;
if (isMockKey) {
    console.warn("[Supabase] Mock key detected in .env. Falling back to local offline mock client.");
    supabaseClient = new MockSupabaseClient();
} else {
    try {
        const realClient = createClient(supabaseUrl, supabaseKey);
        
        // Wrap with a proxy that catches fetch/network errors and falls back to mock client
        const mockClient = new MockSupabaseClient();
        let useMockFallback = false;

        supabaseClient = new Proxy(realClient, {
            get(target, prop) {
                if (useMockFallback) {
                    return mockClient[prop];
                }

                // If auth methods are called, wrap them to catch network failures
                if (prop === 'auth') {
                    return new Proxy(target.auth, {
                        get(authTarget, authProp) {
                            const originalMethod = authTarget[authProp];
                            if (typeof originalMethod === 'function') {
                                return async (...args) => {
                                    try {
                                        const res = await originalMethod.apply(authTarget, args);
                                        // "Failed to fetch" is returned inside error response
                                        if (res?.error?.message === 'Failed to fetch') {
                                            console.warn("[Supabase] Network error detected. Falling back to offline mock client.");
                                            useMockFallback = true;
                                            return mockClient.auth[authProp](...args);
                                        }
                                        return res;
                                    } catch (err) {
                                        if (err.message === 'Failed to fetch') {
                                            console.warn("[Supabase] Network exception caught. Falling back to offline mock client.");
                                            useMockFallback = true;
                                            return mockClient.auth[authProp](...args);
                                        }
                                        throw err;
                                    }
                                };
                            }
                            return originalMethod;
                        }
                    });
                }

                if (prop === 'from') {
                    return (table) => {
                        const realBuilder = target.from(table);
                        return new Proxy(realBuilder, {
                            get(builderTarget, builderProp) {
                                // If they execute the query, catch potential network failures
                                if (builderProp === 'then' || builderProp === 'execute') {
                                    return async (...args) => {
                                        try {
                                            const res = await builderTarget[builderProp](...args);
                                            if (res?.error?.message === 'Failed to fetch') {
                                                console.warn(`[Supabase] Network query failed for table ${table}. Falling back to offline mock client.`);
                                                useMockFallback = true;
                                                return mockClient.from(table)[builderProp](...args);
                                            }
                                            return res;
                                        } catch (err) {
                                            if (err.message === 'Failed to fetch') {
                                                console.warn(`[Supabase] Network query exception for table ${table}. Falling back to offline mock client.`);
                                                useMockFallback = true;
                                                return mockClient.from(table)[builderProp](...args);
                                            }
                                            throw err;
                                        }
                                    };
                                }
                                return builderTarget[builderProp];
                            }
                        });
                    };
                }

                return target[prop];
            }
        });
    } catch (e) {
        console.warn("[Supabase] Failed to initialize real client. Falling back to local offline mock client.", e);
        supabaseClient = new MockSupabaseClient();
    }
}

export default supabaseClient;
