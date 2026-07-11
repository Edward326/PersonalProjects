# WorkHub

## Purpose

WorkHub is a full-stack web application for organization / workforce management: companies
register on the platform, structure themselves into departments and roles, and manage employees,
task assignments, attendance, internal news and events, and performance statistics from a single
dashboard. It is built as a classic three-tier web app — a React single-page frontend, an
Express REST API backend, and a PostgreSQL database — organized as two independently run
projects (`frontend/`, `backend/`) sharing one SQL schema (`database/`).

## Architecture

```
WorkHub/
├── frontend/     React 19 + Vite single-page app
├── backend/      Express REST API (Node, ES modules)
├── database/     PostgreSQL schema and seed data
└── docs/         Entity-relationship documentation (dbSchema.pdf)
```

The frontend and backend are fully decoupled: the frontend talks to the backend only over HTTP
(`fetch`/`axios`), and the backend exposes a versioned-by-resource REST API under `/api/*`,
backed directly by PostgreSQL via the `pg` driver (no ORM).

## Backend

**Stack:** Express, `pg` (PostgreSQL driver), `jsonwebtoken`, `bcryptjs`, `cors`, `morgan`,
`dotenv`.

`src/server.js` wires up global middleware (CORS restricted to the configured frontend origin,
JSON body parsing, request logging in development) and mounts one router per resource under
`/api/<resource>`, plus a `GET /api/health` endpoint and generic 404 / error-handling
middleware.

### Authentication (`middleware/auth_middleware.js`, `routes/auth.js`)
JWT-based auth with three user types — **CEO**, **admin**, and **employee**:
- `generateToken()` signs a 7-day JWT containing `userId`, `userType` and `organizationId`.
- `verifyToken()` is applied as route middleware to protect endpoints, attaching the decoded
  payload to `req.user`.
- `isCEO()` and `belongsToOrganization()` provide additional authorization checks for
  CEO-only and organization-scoped endpoints.
- The login endpoint (`POST /api/auth/login`) branches on whether a `uniqueCode` was supplied:
  with one, it authenticates a CEO against the organization's stored credentials and org code;
  without one, it authenticates a regular user by email/password.

### API resources (`src/routes/`)
Fourteen route modules, each covering one resource area of the schema below:

| Route | Responsibility |
|---|---|
| `organizations.js` | Organization profile, registration, settings |
| `departments.js` | Department CRUD and department-level look/branding config |
| `roles.js` | Role definitions and their permission flags |
| `users.js` | Employee accounts and profiles |
| `auth.js` | Login and JWT issuance |
| `joinRequests.js` | Employees requesting to join an organization/department |
| `assignments.js`, `assignmentUsers.js`, `assignmentOperations.js`, `assignmentSubmissions.js`, `assignmentRelocations.js` | Task lifecycle: creation, assignment to users, status changes, submissions, reassignment between employees |
| `attendance.js`, `departmentAttendanceStats.js` | Attendance tracking and per-department rollups |
| `news.js` | Internal news posts (including a public feed endpoint) |
| `events.js` | Company events/calendar |
| `statistics.js` | Aggregate organization/employee performance statistics |

### Database
`config/database.js` sets up a PostgreSQL connection pool (`pg.Pool`) and exposes a `query()`
helper used throughout the routes, plus `getClient()` for multi-statement transactions.

`database/initial_schema.sql` defines the full relational schema — 15 tables:
`organizations`, `departments`, `roles`, `users`, `join_requests`, `assignments`,
`assignment_users`, `assignment_submissions`, `assignment_relocations`, `news_posts`, `events`,
`attendance`, `department_attendance_stats`, `new_hires_tracking`, `user_statistics` — with
foreign keys cascading appropriately (e.g. deleting an organization cascades to its departments,
users and assignments). Notable design points:
- `roles` carries a set of boolean **permission flags**
  (`can_accept_requests`, `can_post_news`, `can_assign_tasks`, `can_view_statistics`,
  `can_hire`, `can_reassign_tasks`, `can_modify_department_look`, etc.) that drive what a given
  role is allowed to do in the app, rather than a fixed role hierarchy.
- `departments` stores per-department **branding/layout configuration** (header color, logo
  size, sidebar position, hover/selected colors) so each department's view can be visually
  customized.
- `initial_data.sql` provides seed data to populate a fresh database for local development.
- A full entity-relationship diagram is provided in `docs/dbSchema.pdf`.

## Frontend

**Stack:** React 19, React Router 7, Vite 7, Bootstrap 5 + `react-bootstrap-icons`, `axios`,
`zustand` (state management dependency present, not yet wired up — see Notes).

### Routing (`App.jsx`)
- **Public routes:** `/welcome`, `/choose-register`, `/login`, `/employee-register`,
  `/organization-register`.
- **Protected routes** (wrapped in `ProtectedRoute`, which checks for a valid auth token before
  rendering): `/` (Home), `/requests`, `/assignments`, `/organization`, `/statistics`,
  `/settings`.

### Pages (`components/Pages/`)
- **`WelcomePage`** — public landing page (company/news showcase before login).
- **`HomePage`** — the main authenticated dashboard: organization/user/department/role info,
  news feed, and events.
- **`RequestsPage`** — join requests (employees asking to join a department/role) for
  reviewers to approve or reject.
- **`AssignmentPage`** — task/assignment board.
- **`OrganizationPage`** — organization structure browser (departments, org tree, employee
  management).
- **`StatisticsPage`** — organization-, department- and employee-level performance statistics.
- **`SettingsPage`** — account and organization settings.

### Feature sections (`components/Sections/`)
Larger pages are composed from focused section components, including
`AssignmentsSection` / `AssignmentAdminSection` (task lists and admin task management),
`AdministrationSection` (the largest section — organization/role/permission administration),
`OrganizationRegister` (the largest single component — full multi-step organization sign-up
flow), `OrganizationTreeSection` / `OrganizationStatisticsSection`, `ReassignmentSection`,
`JoinRequestsSection`, `EventsSection`, `NewsSection_Home` / `NewsSection_Welcome`,
`TasksStatisticsSection`, `EmployeeStatisticsSection`, `ClassementSection` (leaderboard/ranking),
and `SearchSection`.

### Auth flows (`components/Auth/`)
`LoginPage`, `ChooseRegister` (choose employee vs. organization signup),
`EmployeeRegister` and `OrganizationRegister` (the most substantial component in the codebase,
covering the full organization onboarding form).

### Shared UI (`components/Shared/`)
Reusable, view-specific `Header` and `Sidebar` components (e.g. `Sidebar_Home`,
`Sidebar_Organization`, `Sidebar_Assignments`, `Sidebar_Statistics`, `Sidebar_Settings`), plus
reusable `Cards` (`EmployeeCard`, `CompanyCard`, `EventCard`, `NewsCard`, `RankingTable`) and a
`NewsModal`.

## Notes

- **Frontend data fetching is currently inline, not centralized.** Page/section components call
  the backend directly with hardcoded `fetch('http://localhost:3000/api/...')` calls rather than
  going through `services/*.js`, `hooks/use*.jsx` or `context/*Context.jsx` — those files exist
  as scaffolding for a planned refactor (centralized API services, auth/organization React
  Context, and reusable data-fetching hooks) but are currently empty. `services/api.js` is the
  only service file with real content so far (a small `axios` wrapper for organizations and
  public news). Wiring the rest of the app through these files would remove the hardcoded API
  base URL and reduce duplicated fetch logic across components.
- **Environment configuration:** the backend expects a `.env` file with `PORT`, `DB_HOST`,
  `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `JWT_SECRET`, `FRONTEND_URL` and `NODE_ENV`;
  the frontend reads `VITE_API_URL` (falling back to `http://localhost:3000/api`).
- **Running locally:** create the database with `database/initial_schema.sql` (optionally
  followed by `database/initial_data.sql` for sample data), then run `npm install && npm run dev`
  in `backend/` and `npm install && npm run dev` in `frontend/` (default ports 3000 and 5173).
