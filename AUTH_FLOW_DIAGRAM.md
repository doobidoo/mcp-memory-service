# Authentication Flow Diagram

## Page Load Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Opens Dashboard                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ MemoryDashboard()      │
                    │ Constructor            │
                    │                        │
                    │ Load authState from    │
                    │ localStorage:          │
                    │ - mcp_api_key          │
                    │ - mcp_oauth_token      │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ async init()           │
                    │                        │
                    │ 1. initI18n()          │
                    │ 2. loadSettings()      │
                    │ 3. applyTheme()        │
                    │ 4. setupEventListeners()│
                    │ 5. setupAuthListeners()│
                    │ 6. setupSSE()          │
                    └────────────┬───────────┘
                                 │
                                 ▼
            ┌────────────────────────────────────┐
            │ detectAuthRequirement()            │
            │                                    │
            │ Fetch: /api/health                │
            └────────────┬───────────────────────┘
                         │
         ┌───────────────┴──────────────┐
         │                              │
         ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│ 401 Unauthorized │          │ 200 OK           │
│                  │          │                  │
│ requiresAuth =   │          │ requiresAuth =   │
│ true             │          │ false            │
└────────┬─────────┘          │ isAuthenticated =│
         │                    │ true             │
         │                    │ authMethod =     │
         │                    │ 'anonymous'      │
         │                    └────────┬─────────┘
         │                             │
         ▼                             ▼
┌─────────────────┐          ┌──────────────────┐
│ Has stored      │          │ Continue normal  │
│ credentials?    │          │ initialization   │
│                 │          │                  │
│ apiKey or       │          │ - loadVersion()  │
│ oauthToken?     │          │ - loadDashboard  │
└────┬──────┬─────┘          │   Data()         │
     │      │                │ - checkSync      │
     │      │                │   Status()       │
    No     Yes               └──────────────────┘
     │      │
     │      └─────────────────────┐
     │                            │
     ▼                            ▼
┌─────────────────┐    ┌───────────────────────┐
│ showAuthModal() │    │ Try authentication    │
│                 │    │ with stored creds     │
│ Stop init()     │    │                       │
│ Wait for user   │    │ apiCall('/health')    │
│ authentication  │    │ with headers          │
└─────────────────┘    └───────┬───────────────┘
                               │
                   ┌───────────┴──────────┐
                   │                      │
                   ▼                      ▼
          ┌──────────────┐      ┌──────────────┐
          │ Success      │      │ Fail (401)   │
          │              │      │              │
          │ Continue     │      │ handleAuth   │
          │ initialization│      │ Failure()    │
          └──────────────┘      │              │
                                │ Clear creds  │
                                │ Show modal   │
                                └──────────────┘
```

## Authentication Flow (User Action)

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Authentication Modal Shown                         │
│                                                                      │
│  Option 1: API Key        │        Option 2: OAuth                  │
│  [input field]            │        [OAuth Button]                   │
│  [Authenticate Button]    │                                         │
└────────┬─────────────────┴────────────────────┬───────────────────┘
         │                                       │
         ▼                                       ▼
┌────────────────────────┐           ┌─────────────────────────┐
│ User enters API key    │           │ User clicks OAuth       │
│ Presses Enter or       │           │                         │
│ clicks Authenticate    │           │ Redirect to:            │
└────────┬───────────────┘           │ /oauth/authorize        │
         │                           │                         │
         ▼                           │ (Server handles OAuth   │
┌────────────────────────┐           │  flow)                  │
│ authenticateWithApiKey │           └─────────────────────────┘
│ (apiKey)               │
│                        │
│ 1. Store in            │
│    localStorage        │
│ 2. Test with           │
│    apiCall('/health')  │
└────────┬───────────────┘
         │
         │ Headers:
         │ X-API-Key: {apiKey}
         │
         ▼
    ┌────────────┐
    │ /api/health│
    └────┬───────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ 200 OK │ │ 401    │
└────┬───┘ └────┬───┘
     │          │
     ▼          ▼
┌───────────────────┐  ┌─────────────────────┐
│ SUCCESS           │  │ FAILURE             │
│                   │  │                     │
│ 1. Set authState: │  │ 1. Show error toast │
│    isAuthenticated│  │ 2. Clear apiKey     │
│    = true         │  │ 3. Remove from      │
│    authMethod =   │  │    localStorage     │
│    'api_key'      │  │ 4. Keep modal open  │
│                   │  │                     │
│ 2. Show success   │  └─────────────────────┘
│    toast          │
│                   │
│ 3. closeModal()   │
│                   │
│ 4. loadDashboard  │
│    Data()         │
└───────────────────┘
```

## API Call Flow (with Authentication)

```
┌─────────────────────────────────────────────────────────────────────┐
│              Any API Operation (e.g., Add Memory)                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ apiCall(endpoint,      │
                    │         method, data)  │
                    │                        │
                    │ Build options:         │
                    │ - method               │
                    │ - headers              │
                    │ - body                 │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Add Authentication     │
                    │ Headers:               │
                    │                        │
                    │ If authState.apiKey:   │
                    │   X-API-Key: {key}     │
                    │                        │
                    │ If authState.oauthToken│
                    │   Authorization:       │
                    │   Bearer {token}       │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ fetch(endpoint,        │
                    │       options)         │
                    └────────────┬───────────┘
                                 │
                 ┌───────────────┴──────────────┐
                 │                              │
                 ▼                              ▼
        ┌──────────────┐              ┌──────────────┐
        │ 401          │              │ 200 OK       │
        │ Unauthorized │              │              │
        └────────┬─────┘              │ Return JSON  │
                 │                    │ data         │
                 ▼                    └──────────────┘
        ┌──────────────────┐
        │ handleAuthFailure│
        │                  │
        │ 1. Clear creds   │
        │    from          │
        │    localStorage  │
        │                  │
        │ 2. Set authState │
        │    isAuthenticated│
        │    = false       │
        │                  │
        │ 3. showAuthModal │
        │                  │
        │ 4. Throw error:  │
        │    "Authentication│
        │     required"    │
        └──────────────────┘
```

## State Diagram

```
                    ┌──────────────────┐
                    │   INITIAL        │
                    │   (Page Load)    │
                    └────────┬─────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
          ▼                                     ▼
   ┌──────────────┐                    ┌──────────────┐
   │  ANONYMOUS   │                    │ NEEDS_AUTH   │
   │              │                    │              │
   │ requiresAuth │                    │ requiresAuth │
   │ = false      │                    │ = true       │
   │              │                    │              │
   │ isAuthenticated                  │ isAuthenticated
   │ = true       │                    │ = false      │
   │              │                    │              │
   │ authMethod = │                    └──────┬───────┘
   │ 'anonymous'  │                           │
   └──────────────┘                           │
          │                            ┌──────┴──────┐
          │                            │             │
          │                            ▼             ▼
          │                    ┌────────────┐ ┌────────────┐
          │                    │ Has Stored │ │  No Stored │
          │                    │ Creds      │ │  Creds     │
          │                    └──────┬─────┘ └──────┬─────┘
          │                           │              │
          │                           ▼              ▼
          │                    ┌────────────┐ ┌────────────┐
          │                    │ Test Auth  │ │ Show Modal │
          │                    │            │ │ AWAITING_  │
          │                    │            │ │ AUTH       │
          │                    └──────┬─────┘ └────────────┘
          │                           │              │
          │                    ┌──────┴──────┐       │
          │                    │             │       │
          │                    ▼             ▼       │
          │            ┌──────────┐   ┌──────────┐  │
          │            │ Valid    │   │ Invalid  │  │
          │            └────┬─────┘   └────┬─────┘  │
          │                 │              │        │
          │                 │              └────────┤
          │                 │                       │
          │                 ▼                       ▼
          │         ┌──────────────┐        ┌──────────────┐
          │         │ AUTHENTICATED│        │ AWAITING_AUTH│
          │         │              │        │              │
          └────────▶│ isAuthenticated───────▶ User enters  │
                    │ = true       │        │ credentials  │
                    │              │        │              │
                    │ authMethod = │        │ Click Auth   │
                    │ 'api_key' or │        │ button       │
                    │ 'oauth'      │        └──────────────┘
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Normal       │
                    │ Operations   │
                    │              │
                    │ API calls    │
                    │ include auth │
                    │ headers      │
                    └──────┬───────┘
                           │
                           │ If 401 received
                           ▼
                    ┌──────────────┐
                    │ Return to    │
                    │ AWAITING_AUTH│
                    │              │
                    │ (handleAuth  │
                    │  Failure)    │
                    └──────────────┘
```

## Key Decision Points

1. **On Page Load**: Check if authentication is required by calling `/api/health`

2. **With Stored Credentials**: Automatically attempt authentication, fallback to modal on failure

3. **Without Stored Credentials**: Show authentication modal immediately, don't load data

4. **On 401 Response**: Clear invalid credentials, show authentication modal

5. **After Successful Auth**: Close modal, store credentials, reload dashboard data

## Security Considerations

- API keys stored in `localStorage` (client-side only, no encryption)
- Credentials cleared on authentication failure
- OAuth flow handled server-side (client only does redirect)
- HTTPS recommended for production (not enforced by code)
- No automatic retry on authentication failure (prevents brute force)
